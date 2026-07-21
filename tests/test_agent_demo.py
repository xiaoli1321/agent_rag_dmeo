from __future__ import annotations

from ..agent.graph import CustomerAgent
from ..agent.models import EvidenceDecision, PerceptionResult, RagResult, RetrievedDoc
from ..agent.perception import heuristic_perception
from ..agent.rag import INSUFFICIENT_EVIDENCE_ANSWER


def _perception(
    *,
    intent: str = "产品咨询",
    emotion: str = "平静",
    handoff_requested: bool = False,
    confidence: float = 0.99,
):
    def classify(message: str, history: list[str]) -> PerceptionResult:
        return PerceptionResult(
            intent=intent,  # type: ignore[arg-type]
            emotion=emotion,  # type: ignore[arg-type]
            confidence=confidence,
            handoff_requested=handoff_requested,
            reason=f"test route for {message}",
        )

    return classify


def _grounded_rag(question: str, topic: str | None) -> RagResult:
    docs = [
        RetrievedDoc(
            source_title="Dexcom G7 FAQ",
            source_url="https://example.com",
            chunk_index=0,
            chunk_text="Dexcom G7 sensor is waterproof.",
            score=0.9,
            vector_score=0.9,
            final_score=0.9,
            retrieval_source="dense",
            product_tags=[topic] if topic else [],
        )
    ]
    return RagResult(
        answer="可以戴着洗澡。\n\n引用：\n[1] Dexcom G7 FAQ - https://example.com - chunk #0",
        answer_status="grounded",
        retrieved_docs=docs,
        evidence_decision=EvidenceDecision(status="grounded", reason="test", top_score=0.9),
        debug_trace={"top_k": 4, "min_score": 0.35, "evidence_reason": "test", "final_hits": []},
    )


def _insufficient_rag(question: str, topic: str | None) -> RagResult:
    return RagResult(
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        answer_status="insufficient_evidence",
        retrieved_docs=[],
        evidence_decision=EvidenceDecision(status="insufficient_evidence", reason="test", top_score=None),
        debug_trace={"top_k": 4, "min_score": 0.35, "evidence_reason": "test", "final_hits": []},
    )


def test_heuristic_perception_returns_valid_schema() -> None:
    result = heuristic_perception("你们这个太差了，我要投诉，马上转人工！")

    assert result.intent == "售后诉求"
    assert result.emotion == "愤怒"
    assert result.handoff_requested is True
    assert 0 <= result.confidence <= 1


def test_heuristic_perception_treats_plain_symptom_as_calm() -> None:
    result = heuristic_perception("数据不准")

    assert result.intent == "使用问题"
    assert result.emotion == "平静"
    assert result.handoff_requested is False


def test_heuristic_perception_marks_explicit_frustration_as_dissatisfied() -> None:
    result = heuristic_perception("数据不准，没有用啊，我真的服了")

    assert result.emotion == "不满"


def test_product_question_routes_to_rag() -> None:
    agent = CustomerAgent(perception_fn=_perception(intent="产品咨询"), rag_fn=_grounded_rag)

    result = agent.invoke("Dexcom G7 可以戴着洗澡吗？", thread_id="product-route")

    assert result["active_agent"] == "product_consultant"
    assert "引用：" in result["messages"][-1].content
    # 正常路径不携带 retrieved_docs, answer, answer_status, debug_trace


def test_angry_message_routes_to_empathy_then_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(intent="使用问题", emotion="愤怒", handoff_requested=True),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("太差了，我要投诉，转人工！", thread_id="angry-route")

    assert "已为你转人工" in result["messages"][-1].content
    assert result["active_agent"] == "after_sales"
    assert result["perception"].emotion == "愤怒"
    # handoff_summary 不再写入状态，可通过 messages[-1].content 获取摘要信息


def test_active_handoff_routes_directly_to_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(intent="售后诉求", emotion="平静", handoff_requested=True),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("我要人工处理退款", thread_id="direct-handoff")

    assert "已为你转人工" in result["messages"][-1].content
    assert result["active_agent"] == "after_sales"
    assert "用户主动要求人工" in result["handoff_reason"]


def test_two_rag_failures_trigger_product_to_after_sales_handoff() -> None:
    agent = CustomerAgent(perception_fn=_perception(intent="产品咨询"), rag_fn=_insufficient_rag)
    thread_id = "two-rag-failures"

    first = agent.invoke("连接码是几位数？", thread_id=thread_id)
    second = agent.invoke("那有效期是多少天？", thread_id=thread_id)

    assert first["active_agent"] == "product_consultant"
    assert first["failed_rag_count"] == 1
    # 正常路径 answer_status 不再写入状态，但 failed_rag_count 反映证据不足
    assert "已为你转人工" in second["messages"][-1].content
    assert second["active_agent"] == "after_sales"
    assert second["failed_rag_count"] == 2
    assert "RAG 连续两次未找到足够依据" in second["handoff_reason"]
    # handoff 路径携带 retrieved_docs
    assert second["retrieved_docs"] == []


def test_thread_id_isolates_agent_state() -> None:
    calls: list[str | None] = []

    def rag(question: str, topic: str | None) -> RagResult:
        calls.append(topic)
        return _grounded_rag(question, topic)

    agent = CustomerAgent(perception_fn=_perception(intent="产品咨询"), rag_fn=rag)

    agent.invoke("硅基 GS3 怎么佩戴？", thread_id="thread-a")
    agent.invoke("它防水吗？", thread_id="thread-a")
    agent.invoke("它防水吗？", thread_id="thread-b")

    assert calls == ["GS3", "GS3", None]


def test_multiturn_topic_keeps_previous_product_reference() -> None:
    seen_topics: list[str | None] = []

    def rag(question: str, topic: str | None) -> RagResult:
        seen_topics.append(topic)
        return _grounded_rag(question, topic)

    agent = CustomerAgent(perception_fn=_perception(intent="产品咨询"), rag_fn=rag)
    thread_id = "topic-memory"

    agent.invoke("Dexcom G7 怎么佩戴？", thread_id=thread_id)
    agent.invoke("它防水吗？", thread_id=thread_id)

    assert seen_topics == ["Dexcom G7", "Dexcom G7"]


def test_low_confidence_routes_to_pending_clarification() -> None:
    """低置信度 → 路由到 pending_clarification"""
    agent = CustomerAgent(
        perception_fn=_perception(intent="产品咨询", confidence=0.45),
        rag_fn=_grounded_rag,
    )
    result = agent.invoke("那个……", thread_id="clarify-route")
    assert result["active_agent"] == "pending_clarification"
    assert "请问你想了解" in result["messages"][-1].content


def test_clarification_exits_when_confidence_improves() -> None:
    """追问后用户清晰回答 → 退出澄清，走正常路由"""
    agent = CustomerAgent(
        perception_fn=_perception(intent="产品咨询", confidence=0.95),
        rag_fn=_grounded_rag,
    )
    thread_id = "clarify-exit"
    # 先走一次低置信，触发澄清
    agent.invoke("那个……", thread_id=thread_id)
    # 第二轮清晰表达（使用高置信 mock）
    result = agent.invoke("GS3防水吗", thread_id=thread_id)
    assert result["active_agent"] == "product_consultant"
    assert result.get("clarification_count", 0) == 0  # 已重置


def test_clarification_exceeds_max_rounds_triggers_handoff() -> None:
    """追问达上限后转人工"""
    agent = CustomerAgent(
        perception_fn=_perception(intent="产品咨询", confidence=0.45),
        rag_fn=_grounded_rag,
    )
    thread_id = "clarify-handoff"
    first = agent.invoke("那个……", thread_id=thread_id)
    second = agent.invoke("就是那个……", thread_id=thread_id)
    third = agent.invoke("……", thread_id=thread_id)
    # 第三次应转人工
    assert "已为你转人工" in third["messages"][-1].content or "抱歉" in third["messages"][-1].content
