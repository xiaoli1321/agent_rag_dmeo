from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .models import (
    ActiveAgent,
    AgentState,
    IntentDraft,
    PendingClarification,
    PerceptionResult,
    RagResult,
    RetrievedDoc,
)
from .perception import PerceptionService, decide_perception
from .rag import RagService, RagSubgraphState, build_rag_subgraph
from .run_logger import AgentRunLogger
from ..config import DemoSettings, get_settings


# ── 类型别名 ──────────────────────────────────────────────
# 感知函数签名：输入(用户消息, 历史消息列表)，输出感知结果
PerceptionFn = Callable[[str, list[str]], PerceptionResult]
# RAG 检索函数签名：输入(问题, 话题提示)，输出检索+回答结果
RagFn = Callable[[str, str | None], RagResult]


@dataclass
class CustomerAgent:
    """
    客服 Agent 的主控制器。

    整个项目的核心编排类：
    - 构建 LangGraph 状态机图（感知 → 路由 → 各专业 agent）
    - 对外暴露 invoke() 作为唯一入口
    - 自动记录每次对话的延迟和运行日志
    """

    settings: DemoSettings = field(default_factory=get_settings)
    perception_fn: PerceptionFn | None = None  # 可注入的外部感知函数（方便测试时 mock）
    rag_fn: RagFn | None = None  # 可注入的外部 RAG 函数（方便测试时 mock）
    checkpointer: InMemorySaver = field(
        default_factory=InMemorySaver
    )  # LangGraph 内存检查点，用于多轮对话状态管理

    def __post_init__(self) -> None:
        """初始化各个服务组件并编译 LangGraph"""
        self.perception_service = PerceptionService(settings=self.settings)
        self.rag_service = RagService(settings=self.settings)
        self.rag_subgraph = build_rag_subgraph(self.settings, self.rag_service)
        self.run_logger = AgentRunLogger(
            enabled=self.settings.agent_run_log_enabled,
            log_dir=self.settings.agent_run_log_dir,
        )
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def invoke(self, user_message: str, *, thread_id: str | None = None) -> AgentState:
        """
        唯一对外入口：接收用户消息，走完整个图，返回最终状态。

        - thread_id: 对话线程 ID，同一 ID 的多轮消息共享上下文
        - 自动记录每次调用的延迟到运行日志
        """
        resolved_thread_id = thread_id or "demo-thread"
        config = {"configurable": {"thread_id": resolved_thread_id}}
        started = perf_counter()
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_message)]}, config=config
        )
        latency_ms = int((perf_counter() - started) * 1000)
        self.run_logger.log_turn(
            thread_id=resolved_thread_id,
            user_message=user_message,
            state=result,
            latency_ms=latency_ms,
        )
        return result

    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态机图。

        流程：START → intent_perception 子图 → 路由 → [product_consultant | after_sales | empathy_agent | smalltalk] → END

        路由逻辑在 _active_agent_router 中决定下一步去哪。
        """
        graph = StateGraph(AgentState)
        graph.add_node("intent_perception", self._build_intent_perception_subgraph())
        graph.add_node(
            "product_consultant", self._product_consultant
        )  # 产品咨询节点：RAG 检索+回答
        graph.add_node(
            "after_sales", self._after_sales
        )  # 售后节点：转人工并生成交接摘要
        graph.add_node(
            "empathy_agent", self._empathy_agent
        )  # 情绪安抚节点：先安抚再决定转给谁
        graph.add_node("clarify", self._clarify)  # 澄清节点：每轮只追问一个关键缺口
        graph.add_node("smalltalk", self._smalltalk)  # 闲聊节点：简单问候

        graph.add_edge(START, "intent_perception")
        graph.add_conditional_edges(  # 感知 → 按意图路由到不同 agent
            "intent_perception",
            self._active_agent_router,
            {
                "product_consultant": "product_consultant",
                "after_sales": "after_sales",
                "empathy_agent": "empathy_agent",
                "clarify": "clarify",
                "smalltalk": "smalltalk",
            },
        )
        graph.add_edge("product_consultant", END)  # 产品咨询结束 → 结束
        graph.add_edge("after_sales", END)  # 售后结束 → 结束
        graph.add_edge("empathy_agent", END)  # 情绪安抚结束 → 结束
        graph.add_edge("clarify", END)  # 澄清问题发出后等待用户下一轮
        graph.add_edge("smalltalk", END)  # 闲聊结束 → 结束
        return graph

    def _build_intent_perception_subgraph(self):
        """Encapsulate turn resolution, semantic extraction and slot policy as one reusable subgraph."""
        graph = StateGraph(AgentState)
        graph.add_node("resolve_turn", self._resolve_turn)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("decide_route", self._decide_route)
        graph.add_edge(START, "resolve_turn")
        graph.add_edge("resolve_turn", "classify_intent")
        graph.add_edge("classify_intent", "decide_route")
        graph.add_edge("decide_route", END)
        return graph.compile()

    def draw_mermaid(self) -> str:
        """生成 Mermaid 流程图文本，用于可视化展示"""
        return self.graph.get_graph(xray=True).draw_mermaid()

    def _resolve_turn(self, state: AgentState) -> AgentState:
        """Resolve dialogue state deterministically before semantic classification."""
        message = _last_human_message(state["messages"])
        pending = state.get("pending_clarification")
        relation = "new_request"
        if pending:
            if _looks_like_new_request(message):
                relation = "new_request"
            elif any(token in message for token in ("不是", "不对", "改成", "应该是")):
                relation = "correction"
            else:
                relation = "clarification_answer"
        return {"turn_relation": relation}  # type: ignore[typeddict-item]

    def _classify_intent(self, state: AgentState) -> AgentState:
        message = _last_human_message(state["messages"])
        history = _role_labeled_history(state.get("messages", [])[:-1])
        if self.perception_fn is not None:
            result = self.perception_fn(message, history)
            draft = IntentDraft(
                intent=result.intent,
                emotion=result.emotion,
                confidence=result.confidence,
                handoff_requested=result.handoff_requested,
                secondary_intents=result.secondary_intents,
                entities=result.entities,
                evidence=result.reason,
            )
            source = "injected"
        else:
            draft, source = self.perception_service.classify_draft(
                message, history, current_topic=state.get("current_topic")
            )
        return {
            "intent_draft": draft,
            "injected_perception": result if self.perception_fn is not None else None,
            "perception_trace": {
                "semantic_classification": draft.model_dump(),
                "classifier_source": source,
            },
        }

    def _decide_route(self, state: AgentState) -> AgentState:
        message = _last_human_message(state["messages"])
        pending = state.get("pending_clarification")
        relation = state.get("turn_relation", "new_request")
        draft = state["intent_draft"]
        source = state.get("perception_trace", {}).get("classifier_source", "fallback")
        injected = state.get("injected_perception")
        perception = (
            injected
            if injected is not None
            else decide_perception(
                draft,
                message=message,
                current_topic=state.get("current_topic"),
                pending_clarification=pending,
                turn_relation=relation,
                classifier_source=source,
            )
        )
        if pending and relation == "new_request":
            pending = None
        resolved_message = message
        if pending and relation in {"clarification_answer", "correction"}:
            resolved_message = f"{pending.original_request}\n用户补充：{message}"
        if perception.actionability == "needs_clarification":
            pending = PendingClarification(
                original_request=pending.original_request if pending else message,
                suspected_intent=perception.intent,
                missing_slots=perception.clarification.missing_slots,
                collected_entities=perception.entities,
                turn_count=pending.turn_count if pending else 0,
                asked_slots=pending.asked_slots if pending else [],
            )
        else:
            pending = None
        topic = state.get("current_topic")
        if perception.is_general_query:
            topic = None
        elif perception.actionability == "ready" and perception.intent in {
            "产品咨询",
            "使用问题",
        }:
            topic = perception.entities.product or _resolve_topic_with_rag(
                resolved_message, topic, self.rag_service
            )
        active_agent = _select_active_agent(perception, state.get("active_agent"))
        if perception.actionability == "needs_clarification":
            active_agent = (
                "after_sales"
                if pending
                and pending.turn_count >= self.settings.agent_max_clarification_turns
                else "clarify"
            )
        if perception.intent == "闲聊" or perception.actionability == "unsupported":
            active_agent = "smalltalk"
        trace = dict(state.get("perception_trace", {}))
        trace["policy_decision"] = {
            "turn_relation": relation,
            "actionability": perception.actionability,
            "policy_reason": perception.policy_reason,
            "route": active_agent,
        }
        update: AgentState = {
            "perception": perception,
            "current_topic": topic,
            "active_agent": active_agent,
            "pending_clarification": pending,
            "resolved_user_message": resolved_message,
            "dialogue_status": "ready",
            "perception_trace": trace,
        }
        if perception.actionability == "needs_clarification":
            update["dialogue_status"] = "awaiting_clarification"
            if (
                pending
                and pending.turn_count >= self.settings.agent_max_clarification_turns
            ):
                update["handoff_reason"] = (
                    "连续两轮澄清后仍缺少关键信息，需要人工继续确认。"
                )
        return update

    def _clarify(self, state: AgentState) -> AgentState:
        """每轮只提出一个问题；第二轮才展示有限选项。"""
        perception = state["perception"]
        pending = state.get("pending_clarification")
        if pending is None:
            answer = "请再补充一下你想咨询的产品或具体问题。"
            return _non_rag_update(answer, dialogue_status="awaiting_clarification")

        question = perception.clarification.question or "请再补充一下具体信息。"
        answer = _prepend_empathy(state, question)
        next_turn = pending.turn_count + 1
        visible_clarification = perception.clarification
        if next_turn == 1:
            visible_clarification = visible_clarification.model_copy(
                update={"options": []}
            )
            perception = perception.model_copy(
                update={"clarification": visible_clarification}
            )
        pending = pending.model_copy(
            update={
                "turn_count": next_turn,
                "last_question": question,
                "asked_slots": list(
                    dict.fromkeys(
                        pending.asked_slots + perception.clarification.missing_slots
                    )
                ),
            }
        )
        return {
            **_non_rag_update(answer, dialogue_status="awaiting_clarification"),
            "perception": perception,
            "pending_clarification": pending,
        }

    def _empathy_agent(self, state: AgentState) -> AgentState | Command:
        """
        【情绪安抚节点】：只生成共情上下文，并交给专职节点执行。
        """
        perception = state["perception"]
        user_message = state.get("resolved_user_message") or _last_human_message(
            state["messages"]
        )

        # 调用感知服务动态生成个性化共情话术
        empathy_speech = self.perception_service.generate_empathy(
            user_message,
            intent=perception.intent,
            handoff_requested=perception.handoff_requested,
            issue=perception.entities.issue,
        )

        # A. 涉及售后/退换货/明确要人工 -> 交给售后节点生成统一摘要。
        if perception.handoff_requested or perception.intent == "售后诉求":
            reason = "用户情绪愤怒且涉及售后诉求，已完成动态共情安抚并生成优先交接工单。"
            return Command(goto="after_sales", update={
                "active_agent": "after_sales",
                "empathy_prefix": empathy_speech,
                "handoff_reason": reason,
            })

        # B. 情绪强烈但信息还不够 -> 共情后交给澄清节点追问一个槽位。
        if perception.actionability == "needs_clarification":
            return Command(goto="clarify", update={
                "active_agent": "clarify",
                "empathy_prefix": empathy_speech,
            })

        # C. 产品/使用问题 -> 交给产品节点执行 RAG，不在本节点重复业务逻辑。
        if perception.intent in {"产品咨询", "使用问题"}:
            return Command(goto="product_consultant", update={
                "active_agent": "product_consultant",
                "empathy_prefix": empathy_speech,
            })

        answer = (
            f"{empathy_speech}\n\n您可以补充更多问题细节，或随时回复“转人工”联系专员。"
        )
        return _non_rag_update(answer, dialogue_status="completed")

    def _product_consultant(self, state: AgentState) -> AgentState | Command:
        """
        【产品咨询节点】：
        调用 RAG 服务从知识库检索并生成回答。

        逻辑：
        - 如果 RAG 连续两次返回 insufficient_evidence（证据不足），产品咨询 Agent 主动转售后（after_sales）
        - 成功回答 → 重置 failed_count 为 0
        """
        user_message = state.get("resolved_user_message") or _last_human_message(
            state["messages"]
        )
        if self.rag_fn is not None:
            result = self.rag_fn(user_message, state.get("current_topic"))
        else:
            subgraph_result = self.rag_subgraph.invoke(
                {
                    "question": user_message,
                    "topic_hint": state.get("current_topic"),
                    "rewritten_question": "",
                    "candidates": [],
                    "grades": [],
                    "attempt": 0,
                    "rag_result": None,
                }
            )
            result = subgraph_result["rag_result"]
        logger.info("product_consultant: answer_status=%s", result.answer_status)
        failed_count = state.get("failed_rag_count", 0)
        if result.answer_status == "insufficient_evidence":
            failed_count += 1
        else:
            failed_count = 0
        answer = _prepend_empathy(state, result.answer)
        update = {
            "active_agent": "product_consultant",
            "answer": answer,
            "answer_status": result.answer_status,
            "retrieved_docs": result.retrieved_docs,
            "debug_trace": result.debug_trace,
            "failed_rag_count": failed_count,
            "messages": [AIMessage(content=answer)],
            "dialogue_status": "completed",
            "pending_clarification": None,
            "empathy_prefix": None,
            "handoff_reason": None,
            "handoff_summary": None,
        }
        if result.answer_status == "insufficient_evidence" and failed_count >= 2:
            # 同一 threadId 内连续两次 RAG 找不到依据 → 由产品咨询 Agent 主动转人工
            update["active_agent"] = "after_sales"
            update["handoff_reason"] = (
                "RAG 连续两次未找到足够依据，产品咨询 Agent 主动转交售后/人工。"
            )
            return Command(goto="after_sales", update=update)
        return update

    def _smalltalk(self, state: AgentState) -> AgentState:
        """【闲聊节点】：简单的问候回复，不调用 RAG"""
        perception = state.get("perception")
        if perception and perception.actionability == "unsupported":
            answer = "这个问题不在当前 CGM 客服能力范围内。我可以帮你查询产品信息、排查使用问题或处理售后诉求。"
        else:
            answer = (
                "你好，我可以帮你解答 CGM 动态血糖仪的产品、佩戴、读数和常见使用问题。"
            )
        return _non_rag_update(answer, dialogue_status="completed")

    def _after_sales(self, state: AgentState) -> AgentState:
        """
        【售后节点】：
        生成转人工摘要，包含用户问题、意图/情绪、已尝试的回答、命中的知识库来源等，
        方便人工坐席快速接手。
        """
        reason = state.get("handoff_reason") or _default_handoff_reason(state)
        summary = build_handoff_summary(state, reason=reason)
        answer = _prepend_empathy(state, f"已为你转人工。\n\n{summary}")
        return {
            "active_agent": "after_sales",
            "answer": answer,
            "handoff_reason": reason,
            "handoff_summary": summary,
            "messages": [AIMessage(content=answer)],
            "dialogue_status": "handed_off",
            "pending_clarification": None,
            "empathy_prefix": None,
            "answer_status": None,
            "retrieved_docs": [],
            "debug_trace": {},
        }

    def _active_agent_router(self, state: AgentState) -> str:
        """
        【感知后的路由】：决定感知节点之后应该去哪个 agent。

        逻辑：
        优先级：愤怒 → 人工/售后 → 澄清 → 闲聊/域外 → 产品执行。
        """
        perception = state.get("perception")
        if perception is None:
            return "smalltalk"
        if perception.emotion == "愤怒":
            return "empathy_agent"
        if perception.handoff_requested or perception.intent == "售后诉求":
            return "after_sales"
        if perception.actionability == "needs_clarification":
            pending = state.get("pending_clarification")
            if (
                pending
                and pending.turn_count >= self.settings.agent_max_clarification_turns
            ):
                return "after_sales"
            return "clarify"
        if perception.intent == "闲聊" or perception.actionability == "unsupported":
            return "smalltalk"
        return state.get("active_agent") or "product_consultant"


def build_handoff_summary(state: AgentState, *, reason: str) -> str:
    """
    构建转人工交接摘要。

    包含：
    - 用户最近的问题
    - 当前意图和情绪
    - 已尝试的回答
    - 命中的知识库来源
    - 未解决原因
    - 建议坐席下一步操作
    """
    user_messages = [
        _message_to_text(message)
        for message in state.get("messages", [])
        if isinstance(message, HumanMessage)
    ]
    attempted = [
        _message_to_text(message)
        for message in state.get("messages", [])
        if isinstance(message, AIMessage)
    ]
    docs = state.get("retrieved_docs", [])
    perception = state.get("perception")
    emotion = perception.emotion if perception else "未知"
    intent = perception.intent if perception else "未知"
    sources = ", ".join(doc.source_title for doc in docs[:3]) or "无有效命中"
    pending = state.get("pending_clarification")
    clarification = ""
    if pending:
        clarification = (
            f"\n- 澄清记录：已追问 {pending.turn_count} 轮；"
            f"仍缺少 {', '.join(pending.missing_slots) or '关键信息'}"
        )
    return (
        "会话交接摘要：\n"
        f"- 用户问题：{' / '.join(user_messages[-3:]) or '未记录'}\n"
        f"- 当前意图/情绪：{intent} / {emotion}\n"
        f"- 已尝试回答：{'; '.join(attempted[-2:]) or '尚未自动回答'}\n"
        f"- 命中来源：{sources}\n"
        f"- 未解决原因：{reason}\n"
        f"{clarification}\n"
        "- 建议坐席下一步：核实用户设备型号、订单或售后状态，并给出明确处理时限。"
    )


def _default_handoff_reason(state: AgentState) -> str:
    """
    如果没有显式的手动转交原因，根据状态推断一个默认原因。

    优先级：
    1. 用户主动要求人工
    2. 意图是售后诉求
    3. RAG 连续两次证据不足
    4. 兜底：无法可靠解决
    """
    perception = state.get("perception")
    if perception and perception.handoff_requested:
        return "用户主动要求人工。"
    if perception and perception.intent == "售后诉求":
        return "问题涉及售后或业务系统，当前 Demo 无法自动查询。"
    if state.get("failed_rag_count", 0) >= 2:
        return "RAG 连续两次未找到足够依据，产品咨询 Agent 主动转交售后/人工。"
    return "当前自动流程无法可靠解决。"


# ── 工具函数 ──────────────────────────────────────────────


def _last_human_message(messages: list[BaseMessage]) -> str:
    """从消息列表中提取最后一条用户消息"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _message_to_text(message: BaseMessage) -> str:
    """将 BaseMessage 转换为纯文本"""
    return str(message.content)


def _role_labeled_history(messages: list[BaseMessage]) -> list[str]:
    """Keep roles explicit so the classifier cannot mistake an old answer for user intent."""
    rows: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            rows.append(f"用户：{_message_to_text(message)}")
        elif isinstance(message, AIMessage):
            rows.append(f"助手：{_message_to_text(message)}")
    return rows


def _looks_like_new_request(message: str) -> bool:
    """Only clear pending clarification for an obviously independent request."""
    text = message.strip().lower()
    return any(
        marker in text
        for marker in ("天气", "新闻", "股票", "写代码", "讲故事", "写一首诗")
    )


def _non_rag_update(answer: str, *, dialogue_status: str) -> AgentState:
    """清理上轮 RAG/交接残留，避免 checkpointer 合并出过期状态。"""
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "answer_status": None,
        "retrieved_docs": [],
        "debug_trace": {},
        "handoff_reason": None,
        "handoff_summary": None,
        "empathy_prefix": None,
        "dialogue_status": dialogue_status,  # type: ignore[typeddict-item]
    }


def _prepend_empathy(state: AgentState, answer: str) -> str:
    """Consume the one-turn empathy prefix so later turns do not repeat it."""
    prefix = (state.get("empathy_prefix") or "").strip()
    return f"{prefix}\n\n{answer}" if prefix else answer


def _map_product_tags_to_topic(product_tags: list[str]) -> str | None:
    """仅单一产品标签可以自动成为会话主题；通用资料不能擅自锁定型号。"""
    if len(product_tags) != 1:
        return None
    product = product_tags[0]
    p_lower = product.lower()
    if "手表" in product:
        return "硅基手表"
    if "健康app" in p_lower:
        return "硅基动感健康APP"
    if "dexcom" in p_lower or "g7" in p_lower:
        return "Dexcom G7"
    if "libre" in p_lower or "freestyle" in p_lower:
        return "FreeStyle Libre"
    if "三诺" in p_lower:
        return "三诺爱看 CGM"
    if "硅基" in p_lower or "sibionics" in p_lower:
        return "硅基动感 CGM"
    return product


def _resolve_topic(message: str, existing: str | None) -> str | None:
    """只依据消息中的明确产品关键词解析话题，不触发检索。"""
    lowered = message.lower()
    topics = {
        "gs1 pro": "GS1 Pro",
        "gs1": "GS1",
        "gs3": "GS3",
        "eco": "ECO",
        "metatwin": "MetaTwin",
        "ks3": "KS3",
        "硅基手表": "硅基手表",
        "手表": "硅基手表",
        "硅基动感健康app": "硅基动感健康APP",
        "健康app": "硅基动感健康APP",
        "dexcom": "Dexcom G7",
        "g7": "Dexcom G7",
        "libre": "FreeStyle Libre",
        "三诺": "三诺爱看 CGM",
        "硅基": "硅基动感 CGM",
    }
    for keyword, topic in topics.items():
        if keyword in lowered or keyword in message:
            return topic
    return existing


def _resolve_topic_with_rag(
    message: str, existing: str | None, rag_service: RagService
) -> str | None:
    """
    从用户消息中识别产品话题（通过关键词匹配和 RAG 动态匹配相结合）。

    - 优先显式匹配品牌/型号关键词。
    - 若没有显式匹配，利用 RAG 的无偏检索匹配相关产品。
      如果无偏检索到的文档相关度很高，并且优于在当前锁定话题下的检索评分，则进行话题转移。
    """
    lowered = message.lower()
    topics = {
        "gs1 pro": "GS1 Pro",
        "gs1": "GS1",
        "gs3": "GS3",
        "eco": "ECO",
        "metatwin": "MetaTwin",
        "ks3": "KS3",
        "硅基手表": "硅基手表",
        "手表": "硅基手表",
        "硅基动感健康app": "硅基动感健康APP",
        "健康app": "硅基动感健康APP",
        "dexcom": "Dexcom G7",
        "g7": "Dexcom G7",
        "libre": "FreeStyle Libre",
        "三诺": "三诺爱看 CGM",
        "硅基": "硅基动感 CGM",
    }
    for keyword, topic in topics.items():
        if keyword in lowered or keyword in message:
            return topic

    # 代词没有携带新的型号信息，应优先延续已确认的会话主题；否则无偏检索的
    # 高分文档可能把“它防水吗”从 Dexcom G7 错切换到另一个产品。
    pronouns = {"它", "这个", "那个", "这", "其", "该"}
    if existing and any(pronoun in message for pronoun in pronouns):
        return existing

    # 如果当前没有锁定的话题，且消息中包含指示代词（如“它”、“这个”、“那个”等），则不能盲目通过 RAG 匹配出新话题
    # 因为在没有前文时，“它”是没有指代对象的，强行匹配容易造成幻觉和意图污染
    if not existing:
        if any(pronoun in message for pronoun in pronouns):
            return None

    # RAG 动力路径：通过无前缀的检索寻找强匹配的产品
    unbiased_hits = rag_service.retrieve(message, topic_hint=None)
    if not unbiased_hits:
        return existing

    top_hit = unbiased_hits[0]
    score = top_hit.final_score if top_hit.final_score is not None else top_hit.score
    mapped_product = _map_product_tags_to_topic(top_hit.product_tags)

    # 如果检索到的最相关文档的分数很高，且产品清晰且不同于当前话题
    if score >= 0.5 and mapped_product and mapped_product != existing:
        if not existing:
            return mapped_product

        # 检索当前锁定话题下的分数进行对比
        biased_hits = rag_service.retrieve(message, topic_hint=existing)
        biased_score = 0.0
        if biased_hits:
            b_hit = biased_hits[0]
            biased_score = (
                b_hit.final_score if b_hit.final_score is not None else b_hit.score
            )

        # 如果无偏匹配度大幅优于锁定话题的匹配度，或者锁定话题匹配度过低（低于阈值 0.35）
        if biased_score < 0.35 or score > biased_score + 0.15:
            return mapped_product

    return existing


def _select_active_agent(
    perception: PerceptionResult, existing: ActiveAgent | None
) -> ActiveAgent:
    """
    根据感知结果选择应该激活哪个 agent。

    优先级（高→低）：
    1. 情绪愤怒 → empathy_agent（先安抚）
    2. 要求人工或售后诉求 → after_sales（转人工）
    3. 特殊路由：如果当前在 after_sales，但用户问的是普通产品/使用问题且未要求人工 → 转回 product_consultant
    4. 已有活跃 agent → 沿用，让当前 agent 继续接管会话
    5. 产品咨询或使用问题 → product_consultant（RAG 检索）
    6. 兜底 → product_consultant
    """
    if perception.emotion == "愤怒":
        return "empathy_agent"
    if perception.handoff_requested or perception.intent == "售后诉求":
        return "after_sales"
    # 澄清节点只负责本轮追问；下一轮信息补齐后必须重新进入业务路由。
    if existing == "clarify":
        existing = None
    if existing == "after_sales" and perception.intent in {"产品咨询", "使用问题"}:
        return "product_consultant"
    if existing is not None:
        return existing
    if perception.intent in {"产品咨询", "使用问题"}:
        return "product_consultant"
    return existing or "product_consultant"


def new_thread_id() -> str:
    """生成一个新的对话线程 ID"""
    return f"cgm-demo-{uuid4().hex[:8]}"
