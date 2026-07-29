from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class ConversationStore:
    """JSONL-backed conversation history store.

    Layout:
      data/conversations/<thread_id>.jsonl   — one JSON line per message
      data/conversations/index.json          — fast session listing cache

    Each JSONL line (message):
      {"role":"user","text":"...","timestamp":"..."}
      {"role":"agent","text":"...","meta":[...],"suggestions":[...],"state":{...},"timestamp":"..."}

    The index caches {thread_id -> {title, message_count, created_at, last_updated}}
    for O(1) listing without scanning all JSONL files.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            index = self._read_index()
        sessions = list(index.values())
        sessions.sort(key=lambda s: s.get("last_updated", ""), reverse=True)
        return sessions

    def get_session(self, thread_id: str) -> list[dict[str, Any]] | None:
        path = self._session_path(thread_id)
        if not path.exists():
            return None
        with self._lock:
            raw = path.read_text("utf-8")
        messages: list[dict[str, Any]] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line:
                messages.append(json.loads(line))
        # Patch old entries: agent messages whose text lacks the "引用：" section
        # but have it in state.answer (pre-fix stored raw stream tokens instead
        # of the full answer). Use state.answer as text so references show up.
        for msg in messages:
            if msg.get("role") == "agent":
                text = msg.get("text", "")
                state = msg.get("state")
                if state and "引用" not in text:
                    state_answer = state.get("answer", "")
                    if "引用" in state_answer:
                        msg["text"] = state_answer
        return messages

    def append_turn(
        self,
        thread_id: str,
        user_message: str,
        agent_answer: str,
        agent_state: dict[str, Any] | None = None,
        meta: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        user_entry = {
            "role": "user",
            "text": user_message,
            "timestamp": now,
        }
        agent_entry: dict[str, Any] = {
            "role": "agent",
            "text": agent_answer,
            "timestamp": now,
        }
        if meta:
            agent_entry["meta"] = meta
        if suggestions:
            agent_entry["suggestions"] = suggestions
        if agent_state:
            state = _strip_chunk_text(agent_state)
            agent_entry["state"] = state

        path = self._session_path(thread_id)
        with self._lock:
            # Re-check existence under lock — another thread may have deleted
            # the file between our check and the open() call below.
            is_new = not path.exists()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(user_entry, ensure_ascii=False) + "\n")
                f.write(json.dumps(agent_entry, ensure_ascii=False) + "\n")

            title = (
                (user_message[:20] + "...") if len(user_message) > 20 else user_message
            )
            index = self._read_index()
            if thread_id not in index:
                index[thread_id] = {
                    "thread_id": thread_id,
                    "title": title,
                    "message_count": 0,
                    "created_at": now,
                    "last_updated": now,
                }
            entry = index[thread_id]
            entry["message_count"] = entry.get("message_count", 0) + 2
            entry["last_updated"] = now
            if is_new:
                entry["title"] = title
                entry["created_at"] = now
            self._write_index(index)

    def delete_session(self, thread_id: str) -> bool:
        path = self._session_path(thread_id)
        with self._lock:
            deleted = False
            if path.exists():
                path.unlink()
                deleted = True
            index = self._read_index()
            if thread_id in index:
                del index[thread_id]
                deleted = True
            if deleted:
                self._write_index(index)
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(thread_id: str) -> str:
        safe = re.sub(r"[^\w\-.]", "_", thread_id)
        # Avoid colliding with the index filename
        if safe == "index":
            safe = "index_"
        return safe or "unknown"

    def _session_path(self, thread_id: str) -> Path:
        return self._data_dir / f"{self._sanitize(thread_id)}.jsonl"

    def _index_path(self) -> Path:
        return self._data_dir / "index.json"

    def _read_index(self) -> dict[str, Any]:
        path = self._index_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_index(self, index: dict[str, Any]) -> None:
        path = self._index_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def _strip_chunk_text(state: dict[str, Any]) -> dict[str, Any]:
    """Remove bulky chunk_text from retrieved_docs before persisting."""
    docs = state.get("retrieved_docs")
    if not docs:
        return state
    cleaned = []
    for doc in docs:
        if isinstance(doc, dict):
            d = dict(doc)
            d.pop("chunk_text", None)
            cleaned.append(d)
        else:
            cleaned.append(doc)
    state = dict(state)
    state["retrieved_docs"] = cleaned
    return state
