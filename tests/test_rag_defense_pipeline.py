from __future__ import annotations

from customer_agent_demo.agent.models import RetrievedDoc
from customer_agent_demo.agent.rag import INSUFFICIENT_EVIDENCE_ANSWER, RagService
from customer_agent_demo.config import DemoSettings


def _service() -> RagService:
    return RagService(settings=DemoSettings(agent_top_k=4, agent_min_relevance_score=0.35))


def _doc(text: str, *, score: float = 0.82, title: str = "CGM 参数说明") -> RetrievedDoc:
    return RetrievedDoc(
        source_title=title,
        source_url="https://example.com/cgm",
        chunk_index=0,
        chunk_text=text,
        score=score,
        vector_score=score,
        final_score=score,
        retrieval_source="dense",
    )


def test_c1_grader_blocks_high_score_but_unrelated_parameter_doc() -> None:
    class StubRagService(RagService):
        def retrieve(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
            return [_doc("硅基动感传感器支持 14 天连续监测，防水等级为 IP28。")]

    result = StubRagService(settings=_service().settings).answer("连接码是几位数？")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.answer_status == "insufficient_evidence"
    assert result.retrieved_docs == []
    assert result.evidence_decision
    assert result.evidence_decision.reason == "retrieval_mismatch"
    assert result.debug_trace["candidate_hits"][0]["score"] == 0.82
    assert result.debug_trace["document_grades"][0]["binary_score"] == "no"
    assert result.debug_trace["document_grades"][0]["failure_type"] == "retrieval_mismatch"
    assert [step["name"] for step in result.debug_trace["pipeline_steps"][:3]] == [
        "rewrite_question",
        "retrieve",
        "grade_documents",
    ]


def test_c1_marks_empty_retrieval_as_knowledge_missing() -> None:
    class StubRagService(RagService):
        def retrieve(self, question: str, *, topic_hint: str | None = None) -> list[RetrievedDoc]:
            return []

    result = StubRagService(settings=_service().settings).answer("火星真空环境能不能佩戴？")

    assert result.answer_status == "insufficient_evidence"
    assert result.evidence_decision
    assert result.evidence_decision.reason == "knowledge_missing"
    assert result.debug_trace["candidate_hits"] == []


def test_c1_hallucination_check_rejects_unsupported_numbers() -> None:
    service = _service()
    docs = [_doc("连接码请查看设备包装和 App 引导。")]

    decision = service.check_hallucination(
        "连接码是 6 位数。\n\n引用：\n[1] CGM 参数说明 - https://example.com/cgm - chunk #0",
        docs,
    )

    assert decision.status == "failed"
    assert decision.failure_type == "hallucination"
    assert decision.risky_numbers == ["6 位"]


def test_c1_hallucination_check_rejects_unstable_reference_format() -> None:
    service = _service()
    docs = [_doc("连接码请查看设备包装和 App 引导。")]

    decision = service.check_hallucination("连接码请查看包装。", docs)

    assert decision.status == "failed"
    assert decision.failure_type == "format_unstable"
