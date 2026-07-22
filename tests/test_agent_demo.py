from __future__ import annotations

from ..agent.graph import CustomerAgent
from ..agent.models import (
    ClarificationDecision,
    EvidenceDecision,
    PerceptionResult,
    RagResult,
    RetrievedDoc,
    IntentDraft,
)
from ..agent.perception import decide_perception, heuristic_perception
from ..agent.rag import INSUFFICIENT_EVIDENCE_ANSWER
from ..config import DemoSettings
from ..web import _state_to_response
from ..agent.prompts import load_prompt


def _perception(
    *,
    intent: str = "产品咨询",
    emotion: str = "平静",
    handoff_requested: bool = False,
):
    def classify(message: str, history: list[str]) -> PerceptionResult:
        return PerceptionResult(
            intent=intent,  # type: ignore[arg-type]
            emotion=emotion,  # type: ignore[arg-type]
            confidence=0.99,
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


def _offline_settings() -> DemoSettings:
    return DemoSettings(
        _env_file=None,
        qwen_api_base=None,
        qwen_api_key=None,
        llm_api_base=None,
        llm_api_key=None,
        embedding_api_base=None,
        embedding_api_key=None,
        agent_run_log_enabled=False,
    )


def test_heuristic_perception_returns_valid_schema() -> None:
    result = heuristic_perception("你们这个太差了，我要投诉，马上转人工！")

    assert result.intent == "售后诉求"
    assert result.emotion == "愤怒"
    assert result.handoff_requested is True
    assert 0 <= result.confidence <= 1


def test_all_structured_output_prompts_explicitly_require_json() -> None:
    for prompt_name in (
        "perception.md",
        "rag_rewrite.md",
        "rag_document_grader.md",
        "rag_grounding_grader.md",
    ):
        assert "json" in load_prompt(prompt_name).lower()


def test_rag_structured_prompts_render_without_consuming_json_examples() -> None:
    values = {
        "question": "GS3 蓝牙连接不上",
        "topic_hint": "GS3",
        "rejected_context": "",
        "document": "GS3 蓝牙连接处理步骤。",
        "answer": "请重启手机后再试。",
        "evidence": "处理步骤：重启手机。",
    }

    for prompt_name in (
        "rag_rewrite.md",
        "rag_document_grader.md",
        "rag_grounding_grader.md",
    ):
        rendered = load_prompt(prompt_name).format(**values)
        assert "JSON object" in rendered


def test_intent_draft_normalizes_common_dashscope_json_variants() -> None:
    draft = IntentDraft.model_validate({
        "intent": "troubleshooting", "emotion": "frustrated", "confidence": 0.9,
        "entities": {"product": "GS1", "issue_type": "connection"},
    })

    assert draft.intent == "使用问题"
    assert draft.emotion == "不满"
    assert draft.entities.issue == "connection"


def test_blank_llm_entity_is_a_missing_slot_and_routes_to_clarification() -> None:
    draft = IntentDraft.model_validate({
        "intent": "使用问题",
        "emotion": "平静",
        "confidence": 0.95,
        "entities": {"product": "   ", "issue": "蓝牙连接不上", "requested_action": "排障"},
        "evidence": "蓝牙连接不上",
    })

    result = decide_perception(
        draft,
        message="蓝牙连接不上",
        current_topic=None,
        pending_clarification=None,
        turn_relation="new_request",
        classifier_source="llm",
    )

    assert draft.entities.product is None
    assert result.actionability == "needs_clarification"
    assert result.clarification.missing_slots == ["target_product"]


def test_vague_failure_overrides_llm_aftersales_guess_until_detail_is_collected() -> None:
    draft = IntentDraft.model_validate({
        "intent": "售后诉求",
        "emotion": "平静",
        "confidence": 0.9,
        "entities": {"product": "GS3", "issue": "设备损坏"},
        "evidence": "GS3 坏了",
    })

    result = decide_perception(
        draft,
        message="GS3坏了",
        current_topic=None,
        pending_clarification=None,
        turn_relation="new_request",
        classifier_source="llm",
    )

    assert result.intent == "使用问题"
    assert result.actionability == "needs_clarification"
    assert result.clarification.missing_slots == ["problem_detail"]


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

    assert result["answer_status"] == "grounded"
    assert result["active_agent"] == "product_consultant"
    assert result["debug_trace"]["evidence_reason"] == "test"
    assert "引用：" in result["answer"]
    assert result["retrieved_docs"][0].source_title == "Dexcom G7 FAQ"


def test_angry_message_routes_to_empathy_then_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(intent="使用问题", emotion="愤怒", handoff_requested=True),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("太差了，我要投诉，转人工！", thread_id="angry-route")

    assert "已为你转人工" in result["answer"]
    assert result["active_agent"] == "after_sales"
    assert "会话交接摘要" in result["handoff_summary"]
    assert result["perception"].emotion == "愤怒"


def test_active_handoff_routes_directly_to_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(intent="售后诉求", emotion="平静", handoff_requested=True),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("我要人工处理退款", thread_id="direct-handoff")

    assert "已为你转人工" in result["answer"]
    assert result["active_agent"] == "after_sales"
    assert "用户主动要求人工" in result["handoff_reason"]


def test_two_rag_failures_trigger_product_to_after_sales_handoff() -> None:
    agent = CustomerAgent(perception_fn=_perception(intent="产品咨询"), rag_fn=_insufficient_rag)
    thread_id = "two-rag-failures"

    first = agent.invoke("连接码是几位数？", thread_id=thread_id)
    second = agent.invoke("那有效期是多少天？", thread_id=thread_id)

    assert first["answer_status"] == "insufficient_evidence"
    assert first["active_agent"] == "product_consultant"
    assert first["failed_rag_count"] == 1
    assert "已为你转人工" in second["answer"]
    assert second["active_agent"] == "after_sales"
    assert second["failed_rag_count"] == 2
    assert "RAG 连续两次未找到足够依据" in second["handoff_reason"]


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


def test_ambiguous_reference_requires_clarification_without_topic() -> None:
    result = heuristic_perception("这个怎么用？")

    assert result.intent == "使用问题"
    assert result.actionability == "needs_clarification"
    assert result.clarification.missing_slots == ["reference_target"]


def test_missing_product_exposes_clarify_node_then_resumes_product_consultant() -> None:
    calls: list[tuple[str, str | None]] = []

    def rag(question: str, topic: str | None) -> RagResult:
        calls.append((question, topic))
        return _grounded_rag(question, topic)

    agent = CustomerAgent(settings=_offline_settings(), rag_fn=rag)
    thread_id = "clarify-node-visible"

    first = agent.invoke("蓝牙连接不上", thread_id=thread_id)
    second = agent.invoke("GS3", thread_id=thread_id)

    assert first["active_agent"] == "clarify"
    assert first["perception_trace"]["policy_decision"]["route"] == "clarify"
    assert first["dialogue_status"] == "awaiting_clarification"
    assert second["active_agent"] == "product_consultant"
    assert second["dialogue_status"] == "completed"
    assert calls == [("蓝牙连接不上\n用户补充：GS3", "GS3")]


def test_reference_is_ready_when_current_topic_exists() -> None:
    result = heuristic_perception("它怎么用？", current_topic="GS3")

    assert result.intent == "使用问题"
    assert result.actionability == "ready"
    assert result.entities.product == "GS3"


def test_vague_device_failure_asks_for_problem_detail_without_anger() -> None:
    result = heuristic_perception("GS3坏了")

    assert result.intent == "使用问题"
    assert result.emotion == "平静"
    assert result.actionability == "needs_clarification"
    assert result.clarification.missing_slots == ["problem_detail"]


def test_compound_aftersales_request_keeps_secondary_intent() -> None:
    result = heuristic_perception("G7 防水吗，我的订单怎么还没到？")

    assert result.intent == "售后诉求"
    assert result.actionability == "ready"
    assert "产品咨询" in result.secondary_intents


def test_unrelated_request_is_unsupported_instead_of_clarified() -> None:
    result = heuristic_perception("帮我写一首诗")

    assert result.intent == "闲聊"
    assert result.actionability == "unsupported"
    assert result.clarification.needed is False


def test_medical_emergency_expression_is_out_of_scope_and_never_retrieved() -> None:
    calls: list[str] = []

    def rag(question: str, topic: str | None) -> RagResult:
        calls.append(question)
        return _grounded_rag(question, topic)

    agent = CustomerAgent(settings=_offline_settings(), rag_fn=rag)
    result = agent.invoke("低血糖昏迷了怎么办？", thread_id="medical-boundary")

    assert result["perception"].actionability == "unsupported"
    assert result["active_agent"] == "product_consultant" or result["answer_status"] is None
    assert calls == []


def test_angry_usage_question_keeps_automatic_product_route() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)

    result = agent.invoke("GS3 读数不准，太差了", thread_id="angry-but-solvable")

    assert result["active_agent"] == "product_consultant"
    assert result["answer_status"] == "grounded"


def test_perception_trace_exposes_semantics_and_policy() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    result = agent.invoke("CGM 是什么？", thread_id="perception-trace")

    trace = result["perception_trace"]
    assert trace["semantic_classification"]["intent"] == "产品咨询"
    assert trace["policy_decision"]["route"] == "product_consultant"


def test_perception_schema_rejects_incomplete_clarification() -> None:
    try:
        ClarificationDecision(needed=True, reason="missing_target")
    except ValueError:
        pass
    else:  # pragma: no cover - assertion guard
        raise AssertionError("incomplete clarification must fail validation")


def test_multiturn_clarification_resolves_then_calls_rag_once() -> None:
    calls: list[tuple[str, str | None]] = []

    def rag(question: str, topic: str | None) -> RagResult:
        calls.append((question, topic))
        return _grounded_rag(question, topic)

    agent = CustomerAgent(settings=_offline_settings(), rag_fn=rag)
    thread_id = "clarification-resolves"

    first = agent.invoke("这个怎么用？", thread_id=thread_id)
    second = agent.invoke("GS3", thread_id=thread_id)

    assert first["dialogue_status"] == "awaiting_clarification"
    assert first["answer_status"] is None
    assert first["retrieved_docs"] == []
    assert second["dialogue_status"] == "completed"
    assert second["pending_clarification"] is None
    assert calls == [("这个怎么用？\n用户补充：GS3", "GS3")]


def test_second_clarification_round_exposes_options_then_hands_off() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "clarification-limit"

    first = agent.invoke("这个怎么用？", thread_id=thread_id)
    second = agent.invoke("不清楚", thread_id=thread_id)
    third = agent.invoke("还是不清楚", thread_id=thread_id)

    assert first["perception"].clarification.options == []
    assert second["dialogue_status"] == "awaiting_clarification"
    assert second["perception"].clarification.options
    assert third["dialogue_status"] == "handed_off"
    assert "连续两轮澄清" in third["handoff_reason"]


def test_new_unrelated_topic_cancels_pending_clarification() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "clarification-topic-switch"

    agent.invoke("这个怎么用？", thread_id=thread_id)
    result = agent.invoke("今天天气怎么样？", thread_id=thread_id)

    assert result["dialogue_status"] == "completed"
    assert result["pending_clarification"] is None
    assert result["perception"].actionability == "unsupported"


def test_clarification_is_isolated_by_thread_id() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)

    first = agent.invoke("这个怎么用？", thread_id="clarify-a")
    other = agent.invoke("你好", thread_id="clarify-b")

    assert first["pending_clarification"] is not None
    assert other.get("pending_clarification") is None


def test_chat_response_exposes_clarification_contract() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    state = agent.invoke("这个怎么用？", thread_id="clarification-api")

    payload = _state_to_response(state, thread_id="clarification-api")

    assert payload["dialogue_status"] == "awaiting_clarification"
    assert payload["clarification"]["needed"] is True
    assert payload["clarification"]["missing_slots"] == ["target_product"]
    assert payload["secondary_intents"] == []
