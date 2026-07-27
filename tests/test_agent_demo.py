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
        evidence_decision=EvidenceDecision(
            status="grounded", reason="test", top_score=0.9
        ),
        debug_trace={
            "top_k": 4,
            "min_score": 0.35,
            "evidence_reason": "test",
            "final_hits": [],
        },
    )


def _insufficient_rag(question: str, topic: str | None) -> RagResult:
    return RagResult(
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        answer_status="insufficient_evidence",
        retrieved_docs=[],
        evidence_decision=EvidenceDecision(
            status="insufficient_evidence", reason="test", top_score=None
        ),
        debug_trace={
            "top_k": 4,
            "min_score": 0.35,
            "evidence_reason": "test",
            "final_hits": [],
        },
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


def test_all_structured_output_prompts_omit_redundant_json_format_instructions() -> None:
    for prompt_name in (
        "perception.md",
        "perception.jinja2",
        "rag_rewrite.md",
        "rag_document_grader.md",
        "rag_grounding_grader.md",
    ):
        assert "json object" not in load_prompt(prompt_name).lower()


def test_rag_structured_prompts_render_without_error() -> None:
    values = {
        "question": "GS3 蓝牙连接不上",
        "topic_hint": "GS3",
        "rejected_context": "",
        "document": "GS3 蓝牙连接处理步骤。",
        "answer": "请重启手机后再试。",
        "evidence": "处理步骤：重启手机。",
    }

    assert "GS3 蓝牙连接不上" in load_prompt("rag_rewrite.md").format(**values)
    assert "GS3 蓝牙连接处理步骤" in load_prompt("rag_document_grader.md").format(**values)
    assert "请重启手机后再试" in load_prompt("rag_grounding_grader.md").format(**values)


def test_intent_draft_normalizes_common_dashscope_json_variants() -> None:
    draft = IntentDraft.model_validate(
        {
            "intent": "troubleshooting",
            "emotion": "frustrated",
            "confidence": 0.9,
            "entities": {"product": "GS1", "issue_type": "connection"},
        }
    )

    assert draft.intent == "使用问题"
    assert draft.emotion == "不满"
    assert draft.entities.issue == "connection"


def test_blank_llm_entity_is_a_missing_slot_and_routes_to_clarification() -> None:
    draft = IntentDraft.model_validate(
        {
            "intent": "使用问题",
            "emotion": "平静",
            "confidence": 0.95,
            "entities": {
                "product": "   ",
                "issue": "蓝牙连接不上",
                "requested_action": "排障",
            },
            "evidence": "蓝牙连接不上",
        }
    )

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


def test_vague_failure_overrides_llm_aftersales_guess_until_detail_is_collected() -> (
    None
):
    draft = IntentDraft.model_validate(
        {
            "intent": "售后诉求",
            "emotion": "平静",
            "confidence": 0.9,
            "entities": {"product": "GS3", "issue": "设备损坏"},
            "evidence": "GS3 坏了",
        }
    )

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
    agent = CustomerAgent(
        perception_fn=_perception(intent="产品咨询"), rag_fn=_grounded_rag
    )

    result = agent.invoke("GS3 可以戴着洗澡吗？", thread_id="product-route")

    assert result["answer_status"] == "grounded"
    assert result["active_agent"] == "product_consultant"
    assert result["debug_trace"]["evidence_reason"] == "test"
    assert "引用：" in result["answer"]
    assert result["retrieved_docs"][0].source_title == "Dexcom G7 FAQ"


def test_angry_message_routes_to_empathy_then_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(
            intent="使用问题", emotion="愤怒", handoff_requested=True
        ),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("太差了，我要投诉，转人工！", thread_id="angry-route")

    assert "已为你转人工" in result["answer"]
    assert result["active_agent"] == "after_sales"
    assert "会话交接摘要" in result["handoff_summary"]
    assert result["perception"].emotion == "愤怒"


def test_consecutive_angry_turns_auto_escalates_to_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(
            intent="使用问题", emotion="愤怒", handoff_requested=False
        ),
        rag_fn=_grounded_rag,
    )
    thread_id = "consecutive-angry-route"

    # Turn 1: Angry, but handoff not explicitly requested -> Product consultant with empathy
    first = agent.invoke("GS3 读数不准，太差了！", thread_id=thread_id)
    assert first["active_agent"] == "product_consultant"
    assert first["consecutive_angry_count"] == 1
    assert first["answer_status"] == "grounded"

    # Turn 2: Still angry (comforting failed / 安抚无效), count becomes 2 < 3
    second = agent.invoke("按照你说的搞了还是不行，太垃圾了！", thread_id=thread_id)
    assert second["consecutive_angry_count"] == 2
    assert second["active_agent"] == "product_consultant"

    # Turn 3: 3rd angry turn -> Reaches max threshold 3, auto escalate to after_sales
    third = agent.invoke("气死我了，赶紧给我处理", thread_id=thread_id)
    assert third["active_agent"] == "after_sales"
    assert third["consecutive_angry_count"] == 3
    assert "已为你转人工" in third["answer"]
    assert "判定为安抚无效" in third["handoff_reason"]
    assert third["dialogue_status"] == "handed_off"


def test_calm_message_resets_angry_count() -> None:
    calls = 0

    def _dynamic_perception(msg: str, history: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _perception(intent="使用问题", emotion="愤怒", handoff_requested=False)(msg, history)
        return _perception(intent="使用问题", emotion="平静", handoff_requested=False)(msg, history)

    agent = CustomerAgent(
        perception_fn=_dynamic_perception,
        rag_fn=_grounded_rag,
    )
    thread_id = "reset-angry-count"

    first = agent.invoke("搞什么东西，气死了", thread_id=thread_id)
    assert first["consecutive_angry_count"] == 1

    second = agent.invoke("好吧，那具体的蓝牙连接步骤是什么呢？", thread_id=thread_id)
    assert second["consecutive_angry_count"] == 0
    assert second["active_agent"] == "product_consultant"


def test_active_handoff_routes_directly_to_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(
            intent="售后诉求", emotion="平静", handoff_requested=True
        ),
        rag_fn=_grounded_rag,
    )

    result = agent.invoke("我要人工处理退款", thread_id="direct-handoff")

    assert "已为你转人工" in result["answer"]
    assert result["active_agent"] == "after_sales"
    assert "用户主动要求人工" in result["handoff_reason"]


def test_two_rag_failures_trigger_product_to_after_sales_handoff() -> None:
    agent = CustomerAgent(
        perception_fn=_perception(intent="产品咨询"), rag_fn=_insufficient_rag
    )
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

    agent.invoke("硅基手表 怎么佩戴？", thread_id=thread_id)
    agent.invoke("它防水吗？", thread_id=thread_id)

    assert seen_topics == ["硅基手表", "硅基手表"]


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
    assert (
        result["active_agent"] == "product_consultant"
        or result["answer_status"] is None
    )
    assert calls == []


def test_angry_usage_question_keeps_automatic_product_route() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)

    result = agent.invoke("GS3 读数不准，太差了", thread_id="angry-but-solvable")

    assert result["active_agent"] == "product_consultant"
    assert result["answer_status"] == "grounded"


def test_angry_rag_failures_still_escalate_through_product_node() -> None:
    agent = CustomerAgent(
        settings=_offline_settings(),
        perception_fn=_perception(intent="使用问题", emotion="愤怒"),
        rag_fn=_insufficient_rag,
    )
    thread_id = "angry-rag-escalation"

    first = agent.invoke("GS3 读数不准，太差了", thread_id=thread_id)
    second = agent.invoke("还是不行，太差了", thread_id=thread_id)

    assert first["active_agent"] == "product_consultant"
    assert first["failed_rag_count"] == 1
    assert second["active_agent"] == "after_sales"
    assert (
        "RAG 连续两次未找到足够依据" in second["handoff_reason"]
        or "判定为安抚无效" in second["handoff_reason"]
    )


def test_perception_trace_exposes_semantics_and_policy() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    result = agent.invoke("GS3 是什么？", thread_id="perception-trace")

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


def test_clarification_product_extraction_preserves_original_issue() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-issue-preservation"

    first = agent.invoke("蓝牙连接不上", thread_id=thread_id)
    second = agent.invoke("硅基手表", thread_id=thread_id)

    assert first["dialogue_status"] == "awaiting_clarification"
    assert second["dialogue_status"] == "completed"
    assert second["perception"].entities.product == "硅基手表"
    assert "连接不上" in (second["perception"].entities.issue or "")


def test_followup_turn_preserves_product_entity_in_web_response() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-web-response-topic"

    agent.invoke("蓝牙连接不上", thread_id=thread_id)
    agent.invoke("硅基手表", thread_id=thread_id)
    third = agent.invoke("按步骤操作还是连不上", thread_id=thread_id)

    payload = _state_to_response(third, thread_id=thread_id)

    assert payload["current_topic"] == "硅基手表"
    assert payload["perception"]["entities"]["product"] == "硅基手表"
    assert payload["perception"]["entities"]["issue"] == "连不上"


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


def test_emotional_continuation_inherits_current_issue_without_re_clarifying() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-issue-memory-emotion"

    first = agent.invoke("我这个GS3蓝牙总是连不上", thread_id=thread_id)
    assert first["current_topic"] == "GS3"
    assert first["current_issue"] == "连不上"

    second = agent.invoke("真的没用过这么垃圾的产品了", thread_id=thread_id)
    assert second["current_topic"] == "GS3"
    assert second["active_agent"] != "clarify"
    assert "GS3具体出现了什么问题或表现" not in second["answer"]


def test_vague_complaint_like_nanyong_cleans_issue_and_does_not_deadlock() -> None:
    draft = IntentDraft.model_validate(
        {
            "intent": "使用问题",
            "emotion": "不满",
            "confidence": 0.9,
            "entities": {"product": "GS3", "issue": "难用"},
            "evidence": "GS3太难用了",
        }
    )

    result = decide_perception(
        draft,
        message="GS3太难用了",
        current_topic=None,
        pending_clarification=None,
        turn_relation="new_request",
        classifier_source="llm",
    )

    assert result.entities.issue is None


def test_new_request_refreshes_current_issue_instead_of_stuck_old_issue() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-issue-refresh"

    first = agent.invoke("GS3 蓝牙连接不上", thread_id=thread_id)
    assert first["current_topic"] == "GS3"
    assert first["current_issue"] in ("连接不上", "蓝牙连接不上")

    second = agent.invoke("这个产品具体怎么佩戴？", thread_id=thread_id)
    assert second["current_topic"] == "GS3"
    assert second["perception"].entities.issue == "佩戴"
    assert second["current_issue"] == "佩戴"


def test_troubleshooting_feedback_inherits_current_issue_without_reasking_missing_slot() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-feedback-no-reask"

    first = agent.invoke("GS3 蓝牙连接不上", thread_id=thread_id)
    assert first["current_issue"] in ("连接不上", "蓝牙连接不上")

    second = agent.invoke("按你的操作还是不行啊，真是个烂产品", thread_id=thread_id)
    assert second["dialogue_status"] != "awaiting_clarification"
    assert second["perception"].actionability != "needs_clarification"
    assert "GS3具体出现了什么问题或表现" not in second["answer"]
    assert second["current_issue"] in ("连接不上", "蓝牙连接不上")


def test_consecutive_negative_feedback_triggers_auto_handoff() -> None:
    agent = CustomerAgent(settings=_offline_settings(), rag_fn=_grounded_rag)
    thread_id = "test-auto-handoff"

    agent.invoke("GS3 蓝牙连接不上", thread_id=thread_id)
    second = agent.invoke("按你的操作还是不行啊，真是个烂产品", thread_id=thread_id)
    assert second["consecutive_angry_count"] == 1

    third = agent.invoke("按照你的方法根本没用，服了", thread_id=thread_id)
    assert third["consecutive_angry_count"] == 2

    fourth = agent.invoke("垃圾产品，太差劲了", thread_id=thread_id)
    assert fourth["consecutive_angry_count"] >= 3
    assert fourth["active_agent"] == "after_sales"
    assert fourth["dialogue_status"] == "handed_off"
    assert "转人工" in fourth["answer"]


def test_custom_agent_max_angry_turns_setting() -> None:
    custom_settings = _offline_settings()
    custom_settings.agent_max_angry_turns = 2
    agent = CustomerAgent(settings=custom_settings, rag_fn=_grounded_rag)
    thread_id = "test-custom-angry-turns"

    agent.invoke("GS3 蓝牙连接不上", thread_id=thread_id)
    second = agent.invoke("按你的操作还是不行啊，真是个烂产品", thread_id=thread_id)
    assert second["consecutive_angry_count"] == 1

    third = agent.invoke("按照你的方法根本没用，服了", thread_id=thread_id)
    assert third["consecutive_angry_count"] == 2
    assert third["active_agent"] == "after_sales"
    assert third["dialogue_status"] == "handed_off"



