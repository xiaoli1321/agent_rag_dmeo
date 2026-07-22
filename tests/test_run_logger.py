from __future__ import annotations

import json

from ..agent.models import (
    EvidenceDecision,
    PendingClarification,
    PerceptionResult,
    RetrievedDoc,
)
from ..agent.run_logger import AgentRunLogger, sanitize_record


def _state() -> dict:
    return {
        "perception": PerceptionResult(
            intent="产品咨询",
            emotion="平静",
            confidence=0.9,
            handoff_requested=False,
            reason="test",
        ),
        "active_agent": "product_consultant",
        "current_topic": "Dexcom G7",
        "answer_status": "grounded",
        "retrieved_docs": [
            RetrievedDoc(
                source_title="Dexcom G7 FAQ",
                source_url="https://example.com",
                chunk_index=0,
                chunk_text="waterproof",
                score=0.9,
                vector_score=0.9,
                final_score=0.9,
            )
        ],
        "evidence_decision": EvidenceDecision(status="grounded", reason="test", top_score=0.9),
        "debug_trace": {"qwen_api_key": "secret", "evidence_reason": "test"},
        "handoff_reason": None,
    }


def test_run_logger_writes_jsonl(tmp_path) -> None:
    logger = AgentRunLogger(enabled=True, log_dir=tmp_path)

    path = logger.log_turn(thread_id="t1", user_message="hello", state=_state(), latency_ms=12)

    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["thread_id"] == "t1"
    assert record["user_message"] == "hello"
    assert record["active_agent"] == "product_consultant"
    assert record["latency_ms"] == 12
    assert "qwen_api_key" not in json.dumps(record, ensure_ascii=False)
    assert "intent_draft" in record
    assert "perception_trace" in record


def test_run_logger_disabled_does_not_write(tmp_path) -> None:
    logger = AgentRunLogger(enabled=False, log_dir=tmp_path)

    path = logger.log_turn(thread_id="t1", user_message="hello", state=_state(), latency_ms=12)

    assert path is None
    assert not list(tmp_path.iterdir())


def test_sanitize_record_removes_sensitive_nested_keys() -> None:
    record = sanitize_record({"safe": 1, "nested": {"api_key": "secret", "token_value": "secret"}})

    assert record == {"safe": 1, "nested": {}}


def test_run_logger_records_clarification_state(tmp_path) -> None:
    logger = AgentRunLogger(enabled=True, log_dir=tmp_path)
    state = _state()
    state["dialogue_status"] = "awaiting_clarification"
    state["pending_clarification"] = PendingClarification(
        original_request="这个怎么用？",
        suspected_intent="使用问题",
        missing_slots=["reference_target"],
        turn_count=1,
    )

    path = logger.log_turn(thread_id="clarify", user_message="这个怎么用？", state=state, latency_ms=5)

    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["dialogue_status"] == "awaiting_clarification"
    assert record["clarification_turn_count"] == 1
    assert record["pending_clarification"]["missing_slots"] == ["reference_target"]
