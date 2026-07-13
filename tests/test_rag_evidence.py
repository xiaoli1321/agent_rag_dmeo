from __future__ import annotations

from customer_agent_demo.agent.models import RetrievedDoc
from customer_agent_demo.agent.rag import RagService, _strip_generated_references, dedupe_retrieved_sources, format_references
from customer_agent_demo.config import DemoSettings


def _service() -> RagService:
    return RagService(settings=_test_settings())


def _test_settings() -> DemoSettings:
    return DemoSettings(
        agent_top_k=4, agent_min_relevance_score=0.35, agent_llm_graders_enabled=False,
        qwen_api_base=None, qwen_api_key=None, qwen_llm_model=None,
        llm_api_base=None, llm_api_key=None, llm_model=None,
    )


def _doc(text: str, score: float = 0.8) -> RetrievedDoc:
    return RetrievedDoc(
        source_title="Dexcom G7 FAQ",
        source_url="https://example.com",
        chunk_index=0,
        chunk_text=text,
        score=score,
        vector_score=score,
        final_score=score,
        retrieval_source="dense",
    )


def test_evidence_gate_accepts_pregraded_evidence() -> None:
    service = _service()
    docs = [_doc("Dexcom G7 可在 8 feet 水下最长 24 hours。")]

    decision = service._has_sufficient_evidence("Dexcom G7 可以戴着洗澡多久？", docs)

    assert decision.status == "grounded"
    assert decision.reason == "graded_evidence_available"


def test_evidence_gate_does_not_repeat_document_grading_rules() -> None:
    service = _service()
    docs = [_doc("连接码请查看设备包装和 App 引导。")]

    decision = service._has_sufficient_evidence("连接码是几位数？", docs)

    assert decision.status == "grounded"
    assert decision.reason == "graded_evidence_available"


def test_evidence_gate_trusts_the_prior_document_grader() -> None:
    service = _service()
    docs = [_doc("硅基动感传感器支持 14 天连续监测。")]

    decision = service._has_sufficient_evidence("连接码是几位数？", docs)

    assert decision.status == "grounded"
    assert decision.reason == "graded_evidence_available"


def test_document_grader_rejects_unrelated_evidence_without_product_word_rules() -> None:
    service = _service()
    docs = [_doc("硅基动感传感器支持 14 天连续监测。")]

    decision = service.grade_documents("传感器过期 3 天还能不能继续用？", docs)[0]

    assert decision.binary_score == "no"
    assert decision.failure_type == "retrieval_mismatch"


def test_evidence_gate_rejects_low_score() -> None:
    service = _service()
    docs = [_doc("CGM 是动态血糖监测。", score=0.2)]

    decision = service._has_sufficient_evidence("CGM 是什么？", docs)

    assert decision.status == "insufficient_evidence"
    assert decision.reason.startswith("top_score_below_min_score")


def test_document_grader_rejects_unrelated_context_without_domain_word_rules() -> None:
    service = _service()
    docs = [_doc("设备操作环境为 0°C 至 40°C，存放温度为 -10°C 至 50°C。")]

    decision = service.grade_documents("月球真空环境能不能连续佩戴？", docs)[0]

    assert decision.binary_score == "no"
    assert decision.failure_type == "retrieval_mismatch"


def test_debug_trace_contains_required_fields() -> None:
    service = _service()
    docs = [_doc("CGM 是动态血糖监测。")]
    decision = service._has_sufficient_evidence("CGM 是什么？", docs)

    trace = service._build_debug_trace("CGM 是什么？", docs, decision)

    assert trace["top_k"] == 4
    assert trace["min_score"] == 0.35
    assert trace["evidence_reason"] == "graded_evidence_available"
    assert trace["final_hits"][0]["source_title"] == "Dexcom G7 FAQ"


def test_retrieved_sources_are_deduped_by_source_url() -> None:
    docs = [
        RetrievedDoc(
            source_title="ECO产品介绍-旧",
            source_url="https://example.com/eco",
            chunk_index=0,
            chunk_text="低分片段",
            score=0.2,
            final_score=0.2,
        ),
        RetrievedDoc(
            source_title="ECO产品介绍-旧",
            source_url="https://example.com/eco",
            chunk_index=3,
            chunk_text="高分片段",
            score=0.9,
            final_score=0.9,
        ),
        RetrievedDoc(
            source_title="GS3-佩戴体验",
            source_url="https://example.com/wear",
            chunk_index=1,
            chunk_text="佩戴体验",
            score=0.8,
            final_score=0.8,
        ),
    ]

    deduped = dedupe_retrieved_sources(docs)
    references = format_references(docs)

    assert [doc.source_title for doc in deduped] == ["ECO产品介绍-旧", "GS3-佩戴体验"]
    assert deduped[0].chunk_index == 3
    assert references.count("ECO产品介绍-旧") == 1


def test_retrieved_sources_are_deduped_by_url_even_when_titles_differ() -> None:
    docs = [
        RetrievedDoc(
            source_title="ECO产品介绍-旧",
            source_url="https://example.com/eco",
            chunk_index=0,
            chunk_text="低分片段",
            score=0.2,
            final_score=0.2,
        ),
        RetrievedDoc(
            source_title="ECO 产品介绍 - 旧",
            source_url="https://example.com/eco",
            chunk_index=3,
            chunk_text="高分片段",
            score=0.9,
            final_score=0.9,
        ),
    ]

    deduped = dedupe_retrieved_sources(docs)
    references = format_references(docs)

    assert len(deduped) == 1
    assert deduped[0].source_title == "ECO 产品介绍 - 旧"
    assert references.count("https://example.com/eco") == 1


def test_generated_reference_sections_are_stripped_before_system_references() -> None:
    answer = """可以按说明处理。

引用列表：
[1] ECO 产品介绍 - 旧 - https://example.com/eco - chunk #3
[2] GS3-佩戴体验 - https://example.com/wear - chunk #1"""

    assert _strip_generated_references(answer) == "可以按说明处理。"


def test_generated_trailing_reference_lines_are_stripped_without_header() -> None:
    answer = """可以按说明处理。
[1] ECO 产品介绍 - 旧 - https://example.com/eco - chunk #3
[2] GS3-佩戴体验 - https://example.com/wear - chunk #1"""

    assert _strip_generated_references(answer) == "可以按说明处理。"


def test_insufficient_answer_does_not_return_user_references() -> None:
    class StubRagService(RagService):
        def retrieve(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
            return [_doc("CGM 是动态血糖监测。", score=0.1)]

    service = StubRagService(settings=_test_settings())

    result = service.answer("CGM 是什么？")

    assert result.answer_status == "insufficient_evidence"
    assert "引用：" not in result.answer
    assert result.retrieved_docs == []
    assert result.debug_trace["final_hits"] == []
    assert result.debug_trace["candidate_hits"][0]["source_title"] == "Dexcom G7 FAQ"


def test_llm_refusal_after_grounded_retrieval_keeps_retrieved_docs() -> None:
    class StubRagService(RagService):
        def retrieve(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
            return [_doc("数据不准时，先判断佩戴是否未满 48 小时。", score=0.8)]

        def _generate_answer(self, question: str, docs: list[RetrievedDoc]) -> str:
            return "没有在当前知识库找到足够依据。"

    service = StubRagService(settings=_test_settings())

    result = service.answer("数据不准")

    assert result.answer_status == "grounded"
    assert result.retrieved_docs[0].source_title == "Dexcom G7 FAQ"
    assert result.debug_trace["generation_warning"] == "llm_refused_after_grounded_retrieval"
    assert result.debug_trace["final_hits"][0]["source_title"] == "Dexcom G7 FAQ"
    assert "引用：" in result.answer


def test_resolve_topic_handles_watch_and_patch() -> None:
    from customer_agent_demo.agent.graph import _resolve_topic

    class MockRagService(RagService):
        def retrieve(self, question: str, topic_hint: str | None = None) -> list[RetrievedDoc]:
            if topic_hint == "Dexcom G7":
                return []
            if any(w in question.lower() for w in ("手表", "watch", "加固贴")):
                doc = _doc("硅基手表/加固贴的相关内容", score=0.8)
                doc.product = "硅基动感 CGM"
                return [doc]
            return []

    service = MockRagService(settings=_test_settings())
    assert _resolve_topic("手表什么时候上线", None, service) == "硅基动感 CGM"
    assert _resolve_topic("watch functions", None, service) == "硅基动感 CGM"
    assert _resolve_topic("硅基加固贴尺寸", None, service) == "硅基动感 CGM"
    assert _resolve_topic("加固贴", "Dexcom G7", service) == "硅基动感 CGM"


def test_keyword_overlap_robustness() -> None:
    from customer_agent_demo.agent.rag import _keyword_overlap
    assert _keyword_overlap("硅基加固贴尺寸是多少", "ECO加固贴尺寸 GS1加固贴尺寸") > 0
    assert _keyword_overlap("Dexcom G7 可以戴着洗澡吗？", "Dexcom says the G7 sensor is waterproof.") > 0


def test_hallucination_check_year_mapping() -> None:
    service = _service()
    docs = [_doc("23年10月前的订单发货是一代手表")]
    answer = "2023年10月前的订单发货是一代手表\n引用：\n[1] Dexcom G7 FAQ - https://example.com - chunk #0"

    decision = service.check_hallucination(answer, docs)
    assert decision.status == "grounded"


def test_hallucination_check_unit_spacing() -> None:
    service = _service()
    docs = [_doc("建议餐2在血糖平稳时测两组数据")]
    answer = "建议在餐后 2 小时测血糖。\n引用：\n[1] Dexcom G7 FAQ - https://example.com - chunk #0"

    decision = service.check_hallucination(answer, docs)
    assert decision.status == "grounded"


def test_hallucination_check_catches_actual_hallucination() -> None:
    service = _service()
    docs = [_doc("23年10月前的订单发货是一代手表")]
    answer = "580元是二代手表的售价。\n引用：\n[1] Dexcom G7 FAQ - https://example.com - chunk #0"

    decision = service.check_hallucination(answer, docs)
    assert decision.status == "failed"
    assert decision.failure_type == "hallucination"


def test_strip_generated_references_with_doc_titles() -> None:
    answer = "测试答案。\n[1] 硅基手表故障场景及处理"
    doc1 = _doc("一些测试文本")
    doc1.source_title = "硅基手表故障场景及处理"
    assert _strip_generated_references(answer, [doc1]) == "测试答案。"

    # Test spacing and dash differences (e.g. ECO 产品介绍 - 旧 vs ECO产品介绍-旧)
    answer2 = "测试答案2。\n[1] ECO 产品介绍 - 旧"
    doc2 = _doc("一些测试文本")
    doc2.source_title = "ECO产品介绍-旧"
    assert _strip_generated_references(answer2, [doc2]) == "测试答案2。"
