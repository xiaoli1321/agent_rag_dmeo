from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from customer_agent_demo.agent.models import ActiveAgent, AgentState, PerceptionResult, RagResult, RetrievedDoc
from customer_agent_demo.agent.perception import PerceptionService
from customer_agent_demo.agent.rag import RagService
from customer_agent_demo.agent.run_logger import AgentRunLogger
from customer_agent_demo.config import DemoSettings, get_settings


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
    perception_fn: PerceptionFn | None = None      # 可注入的外部感知函数（方便测试时 mock）
    rag_fn: RagFn | None = None                     # 可注入的外部 RAG 函数（方便测试时 mock）
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)  # LangGraph 内存检查点，用于多轮对话状态管理

    def __post_init__(self) -> None:
        """初始化各个服务组件并编译 LangGraph"""
        self.perception_service = PerceptionService(settings=self.settings)
        self.rag_service = RagService(settings=self.settings)
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
        result = self.graph.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
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

        流程：START → perceive(感知) → 路由 → [product_consultant | after_sales | empathy_agent | smalltalk] → END

        路由逻辑在 _active_agent_router 中决定下一步去哪。
        """
        graph = StateGraph(AgentState)
        graph.add_node("perceive", self._perceive)                          # 感知节点：识别意图和情绪
        graph.add_node("product_consultant", self._product_consultant)      # 产品咨询节点：RAG 检索+回答
        graph.add_node("after_sales", self._after_sales)                    # 售后节点：转人工并生成交接摘要
        graph.add_node("empathy_agent", self._empathy_agent)                # 情绪安抚节点：先安抚再决定转给谁
        graph.add_node("smalltalk", self._smalltalk)                        # 闲聊节点：简单问候

        graph.add_edge(START, "perceive")                                   # 起点 → 感知
        graph.add_conditional_edges(                                        # 感知 → 按意图路由到不同 agent
            "perceive",
            self._active_agent_router,
            {
                "product_consultant": "product_consultant",
                "after_sales": "after_sales",
                "empathy_agent": "empathy_agent",
                "smalltalk": "smalltalk",
            },
        )
        graph.add_edge("product_consultant", END)   # 产品咨询结束 → 结束
        graph.add_edge("after_sales", END)          # 售后结束 → 结束
        graph.add_edge("empathy_agent", END)        # 情绪安抚结束 → 结束
        graph.add_edge("smalltalk", END)            # 闲聊结束 → 结束
        return graph

    def draw_mermaid(self) -> str:
        """生成 Mermaid 流程图文本，用于可视化展示"""
        return self.graph.get_graph().draw_mermaid()

    def _perceive(self, state: AgentState) -> AgentState:
        """
        【感知节点】：
        1. 提取用户最新一条消息
        2. 调用 perception_service 进行意图分类和情绪识别
        3. 解析产品话题（如"硅基"→ 硅基动感 CGM）
        4. 根据感知结果决定当前应激活哪个 agent
        """
        user_message = _last_human_message(state["messages"])
        history = [_message_to_text(message) for message in state.get("messages", [])[:-1]]
        classify = self.perception_fn or self.perception_service.classify
        perception = classify(user_message, history)
        topic = _resolve_topic(user_message, state.get("current_topic"), self.rag_service)
        active_agent = _select_active_agent(perception, state.get("active_agent"))
        return {"perception": perception, "current_topic": topic, "active_agent": active_agent}

    def _empathy_agent(self, state: AgentState) -> Command:
        """
        【情绪安抚节点】：
        用户情绪为"愤怒"时进入此节点。

        逻辑：
        - 先输出一段安抚话术
        - 如果用户要求人工、属于售后诉求、或情绪愤怒 → 转售后（after_sales）
        - 如果只是产品咨询或使用问题 → 转产品咨询（product_consultant）
        - 否则结束对话

        注意：这里用了 Command(goto=..., update=...) 实现节点间跳转，
        这是 LangGraph 的 swarm 模式 —— agent 之间平级，没有中心调度器。
        """
        perception = state["perception"]
        user_message = _last_human_message(state["messages"])
        answer = (
            "我理解这件事给你带来了很差的体验，我先帮你把情况记录清楚。"
            "为了避免继续耽误你，我会根据当前问题判断是否需要人工接手。"
        )
        if perception.handoff_requested:
            answer += " 你已经明确要求人工，我会整理交接摘要。"
        update = {
            "active_agent": "empathy_agent",
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "attempted_answers": [answer],
            "handoff_reason": "用户情绪愤怒，需要先安抚。" if "投诉" in user_message else None,
        }
        if perception.handoff_requested or perception.intent == "售后诉求" or perception.emotion == "愤怒":
            # 需要售后或人工的场景 → 跳转到 after_sales
            update["handoff_reason"] = update["handoff_reason"] or "用户情绪愤怒，需要售后或人工接手。"
            update["active_agent"] = "after_sales"
            return Command(goto="after_sales", update=update)
        if perception.intent in {"产品咨询", "使用问题"}:
            # 普通产品问题 → 跳转到 product_consultant
            update["active_agent"] = "product_consultant"
            return Command(goto="product_consultant", update=update)
        # 其他情况结束
        return Command(goto=END, update=update)

    def _product_consultant(self, state: AgentState) -> AgentState | Command:
        """
        【产品咨询节点】：
        调用 RAG 服务从知识库检索并生成回答。

        逻辑：
        - 如果 RAG 连续两次返回 insufficient_evidence（证据不足），产品咨询 Agent 主动转售后（after_sales）
        - 成功回答 → 重置 failed_count 为 0
        """
        user_message = _last_human_message(state["messages"])
        if self.rag_fn is not None:
            result = self.rag_fn(user_message, state.get("current_topic"))
        else:
            result = self.rag_service.answer(user_message, topic_hint=state.get("current_topic"))
        failed_count = state.get("failed_rag_count", 0)
        if result.answer_status == "insufficient_evidence":
            failed_count += 1
        else:
            failed_count = 0
        update = {
            "active_agent": "product_consultant",
            "answer": result.answer,
            "answer_status": result.answer_status,
            "retrieved_docs": result.retrieved_docs,
            "evidence_decision": result.evidence_decision,
            "debug_trace": result.debug_trace,
            "failed_rag_count": failed_count,
            "messages": [AIMessage(content=result.answer)],
            "attempted_answers": [result.answer],
        }
        if result.answer_status == "insufficient_evidence" and failed_count >= 2:
            # 同一 threadId 内连续两次 RAG 找不到依据 → 由产品咨询 Agent 主动转人工
            update["active_agent"] = "after_sales"
            update["handoff_reason"] = "RAG 连续两次未找到足够依据，产品咨询 Agent 主动转交售后/人工。"
            return Command(goto="after_sales", update=update)
        return update

    def _smalltalk(self, state: AgentState) -> AgentState:
        """【闲聊节点】：简单的问候回复，不调用 RAG"""
        answer = "你好，我可以帮你解答 CGM 动态血糖仪的产品、佩戴、读数和常见使用问题。"
        return {"answer": answer, "messages": [AIMessage(content=answer)], "attempted_answers": [answer]}

    def _after_sales(self, state: AgentState) -> AgentState:
        """
        【售后节点】：
        生成转人工摘要，包含用户问题、意图/情绪、已尝试的回答、命中的知识库来源等，
        方便人工坐席快速接手。
        """
        reason = state.get("handoff_reason") or _default_handoff_reason(state)
        summary = build_handoff_summary(state, reason=reason)
        answer = f"已为你转人工。\n\n{summary}"
        return {
            "active_agent": "after_sales",
            "answer": answer,
            "handoff_reason": reason,
            "handoff_summary": summary,
            "messages": [AIMessage(content=answer)],
        }

    @staticmethod
    def _active_agent_router(state: AgentState) -> str:
        """
        【感知后的路由】：决定感知节点之后应该去哪个 agent。

        逻辑：
        - 如果感知到意图是"闲聊" → 走 smalltalk
        - 否则根据感知阶段选定的 active_agent 路由
        """
        perception = state.get("perception")
        if perception and perception.intent == "闲聊":
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
    user_messages = [_message_to_text(message) for message in state.get("messages", []) if isinstance(message, HumanMessage)]
    attempted = state.get("attempted_answers", [])
    docs = state.get("retrieved_docs", [])
    perception = state.get("perception")
    emotion = perception.emotion if perception else "未知"
    intent = perception.intent if perception else "未知"
    sources = ", ".join(f"{doc.source_title} chunk #{doc.chunk_index}" for doc in docs[:3]) or "无有效命中"
    return (
        "会话交接摘要：\n"
        f"- 用户问题：{' / '.join(user_messages[-3:]) or '未记录'}\n"
        f"- 当前意图/情绪：{intent} / {emotion}\n"
        f"- 已尝试回答：{'; '.join(attempted[-2:]) or '尚未自动回答'}\n"
        f"- 命中来源：{sources}\n"
        f"- 未解决原因：{reason}\n"
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


def _map_product_to_topic(product: str | None) -> str | None:
    if not product:
        return None
    p_lower = product.lower()
    if "dexcom" in p_lower or "g7" in p_lower:
        return "Dexcom G7"
    if "libre" in p_lower or "freestyle" in p_lower:
        return "FreeStyle Libre"
    if "三诺" in p_lower:
        return "三诺爱看 CGM"
    if "硅基" in p_lower or "sibionics" in p_lower:
        return "硅基动感 CGM"
    return product


def _resolve_topic(message: str, existing: str | None, rag_service: RagService) -> str | None:
    """
    从用户消息中识别产品话题（通过关键词匹配和 RAG 动态匹配相结合）。

    - 优先显式匹配品牌/型号关键词。
    - 若没有显式匹配，利用 RAG 的无偏检索匹配相关产品。
      如果无偏检索到的文档相关度很高，并且优于在当前锁定话题下的检索评分，则进行话题转移。
    """
    lowered = message.lower()
    topics = {
        "dexcom": "Dexcom G7",
        "g7": "Dexcom G7",
        "libre": "FreeStyle Libre",
        "三诺": "三诺爱看 CGM",
        "硅基": "硅基动感 CGM",
    }
    for keyword, topic in topics.items():
        if keyword in lowered or keyword in message:
            return topic

    # 如果当前没有锁定的话题，且消息中包含指示代词（如“它”、“这个”、“那个”等），则不能盲目通过 RAG 匹配出新话题
    # 因为在没有前文时，“它”是没有指代对象的，强行匹配容易造成幻觉和意图污染
    if not existing:
        pronouns = {"它", "这个", "那个", "这", "其", "该"}
        if any(p in message for p in pronouns):
            return None

    # RAG 动力路径：通过无前缀的检索寻找强匹配的产品
    unbiased_hits = rag_service.retrieve(message, topic_hint=None)
    if not unbiased_hits:
        return existing

    top_hit = unbiased_hits[0]
    score = top_hit.final_score if top_hit.final_score is not None else top_hit.score
    mapped_product = _map_product_to_topic(top_hit.product)

    # 如果检索到的最相关文档的分数很高，且产品清晰且不同于当前话题
    if score >= 0.5 and mapped_product and mapped_product != existing:
        if not existing:
            return mapped_product

        # 检索当前锁定话题下的分数进行对比
        biased_hits = rag_service.retrieve(message, topic_hint=existing)
        biased_score = 0.0
        if biased_hits:
            b_hit = biased_hits[0]
            biased_score = b_hit.final_score if b_hit.final_score is not None else b_hit.score

        # 如果无偏匹配度大幅优于锁定话题的匹配度，或者锁定话题匹配度过低（低于阈值 0.35）
        if biased_score < 0.35 or score > biased_score + 0.15:
            return mapped_product

    return existing


def _select_active_agent(perception: PerceptionResult, existing: ActiveAgent | None) -> ActiveAgent:
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
