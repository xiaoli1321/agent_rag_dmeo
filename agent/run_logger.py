from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import AgentState


SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "token",
    "secret",
    "password",
    "qwen_api_key",
}


@dataclass(slots=True)
class AgentRunLogger:
    enabled: bool
    log_dir: Path

    def log_turn(
        self,
        *,
        thread_id: str,
        user_message: str,
        state: AgentState,
        latency_ms: int,
    ) -> Path | None:
        if not self.enabled:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"agent_runs_{datetime.now().strftime('%Y%m%d')}.jsonl"
        record = sanitize_record(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "thread_id": thread_id,
                "user_message": user_message,
                "perception": _dump_model(state.get("perception")),
                "intent_draft": _dump_model(state.get("intent_draft")),
                "perception_trace": state.get("perception_trace") or {},
                "active_agent": state.get("active_agent"),
                "current_topic": state.get("current_topic"),
                "answer_status": state.get("answer_status"),
                "dialogue_status": state.get("dialogue_status"),
                "pending_clarification": _dump_model(state.get("pending_clarification")),
                "clarification_turn_count": (
                    state["pending_clarification"].turn_count
                    if state.get("pending_clarification")
                    else 0
                ),
                "secondary_intents": (
                    state["perception"].secondary_intents
                    if state.get("perception")
                    else []
                ),
                "retrieved_docs": [
                    _dump_model(doc) for doc in state.get("retrieved_docs", [])
                ],
                "debug_trace": state.get("debug_trace") or {},
                "handoff_reason": state.get("handoff_reason"),
                "latency_ms": latency_ms,
            }
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path


def sanitize_record(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                continue
            sanitized[key] = sanitize_record(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_record(item) for item in value]
    return value


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(sensitive in lowered for sensitive in SENSITIVE_KEYS)
