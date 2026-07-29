from __future__ import annotations

import argparse
import json
import logging
import pathlib
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any

# Suppress LangSmith missing-API-key warning. Must be before any import that
# transitively loads langsmith, because it fires on first client instantiation.
warnings.filterwarnings(
    "ignore", message="API key must be provided when using hosted LangSmith API"
)

from .agent.conversation_store import ConversationStore
from .agent.graph import CustomerAgent
from .agent.models import PerceptionResult, RetrievedDoc
from .config import DEMO_ROOT, DemoSettings, get_settings

LOGGER = logging.getLogger(__name__)

STATIC_DIR = pathlib.Path(__file__).parent / "static"


def _read_index_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


HTML_PAGE = _read_index_html()


def create_handler(agent: CustomerAgent) -> type[BaseHTTPRequestHandler]:
    store = ConversationStore(DEMO_ROOT / "data" / "conversations")

    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "CustomerAgentDemo/0.1"

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_text(
                    _read_index_html(), content_type="text/html; charset=utf-8"
                )
                return
            if self.path == "/api/health":
                self._send_json({"ok": True})
                return
            if self.path == "/api/conversations":
                self._send_json({"sessions": store.list_sessions()})
                return
            if self.path.startswith("/api/conversations/"):
                thread_id = self.path[len("/api/conversations/") :]
                if not thread_id:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                messages = store.get_session(thread_id)
                if messages is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"thread_id": thread_id, "messages": messages})
                return
            # 静态文件
            static_path = STATIC_DIR / self.path.lstrip("/")
            if static_path.is_file() and static_path.parent == STATIC_DIR:
                ext = static_path.suffix.lstrip(".").lower()
                text_types = {"css", "js", "html", "svg", "json", "xml", "txt"}
                bin_types = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "webp": "image/webp",
                    "ico": "image/x-icon",
                    "svg": "image/svg+xml",
                }
                ctype = bin_types.get(ext) or {
                    "css": "text/css; charset=utf-8",
                    "js": "application/javascript; charset=utf-8",
                    "html": "text/html; charset=utf-8",
                }.get(ext, "application/octet-stream")
                if ext in text_types:
                    self._send_text(
                        static_path.read_text(encoding="utf-8"), content_type=ctype
                    )
                else:
                    self._send_bytes(static_path.read_bytes(), content_type=ctype)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/chat/stream":
                self._handle_chat_stream()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            if self.path.startswith("/api/conversations/"):
                thread_id = self.path[len("/api/conversations/") :]
                if not thread_id:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                store.delete_session(thread_id)
                self._send_json({"ok": True})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_chat_stream(self) -> None:
            try:
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                thread_id = str(
                    payload.get("thread_id") or "web-default-thread"
                ).strip()
                if not message:
                    self._send_sse_event("error", {"error": "message is required"})
                    return

                LOGGER.info(
                    "stream_chat: message=%s thread=%s", message[:60], thread_id
                )
                self._send_sse_headers()

                answer = ""
                state = None

                for event_type, data in agent.stream_chat(message, thread_id=thread_id):
                    if event_type == "answer_token" and isinstance(data, str):
                        answer += data
                        self._send_sse_event("answer_token", {"token": data})
                    elif event_type == "state" and isinstance(data, dict):
                        state = data
                    elif event_type == "error" and isinstance(data, str):
                        self._send_sse_event("error", {"error": data})

                response = (
                    _state_to_response(
                        state, thread_id=thread_id, settings=agent.settings
                    )
                    if state
                    else {}
                )
                self._send_sse_event("state", response)
                self._send_sse_event("done", {})

                # Persist conversation after streaming is done
                if state:
                    try:
                        meta, suggestions = _extract_meta_and_suggestions(response)
                        store.append_turn(
                            thread_id,
                            message,
                            response.get("answer") or answer,
                            response,
                            meta=meta,
                            suggestions=suggestions,
                        )
                    except Exception:
                        LOGGER.exception("Failed to persist conversation")
            except Exception as exc:
                LOGGER.exception("stream chat failed")
                try:
                    self._send_sse_event("error", {"error": str(exc)})
                    self._send_sse_event("done", {})
                except Exception:
                    pass

        def _send_sse_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

        def _send_sse_event(self, event: str, data: dict[str, Any]) -> None:
            line = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(line.encode("utf-8"))
            self.wfile.flush()

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data: bytes, *, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)

    return DemoRequestHandler


def _state_to_response(
    state: dict[str, Any],
    *,
    thread_id: str,
    settings: DemoSettings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or get_settings()
    perception = state.get("perception")
    docs = state.get("retrieved_docs") or []
    current_topic = state.get("current_topic")
    perception_dict = _model_to_dict(perception)
    if perception_dict and isinstance(perception_dict, dict) and current_topic:
        entities = perception_dict.setdefault("entities", {})
        if isinstance(entities, dict) and not entities.get("product"):
            entities["product"] = current_topic
    return {
        "thread_id": thread_id,
        "answer": state.get("answer") or "",
        "active_agent": state.get("active_agent"),
        "answer_status": state.get("answer_status"),
        "dialogue_status": state.get("dialogue_status"),
        "handoff_reason": state.get("handoff_reason"),
        "handoff_summary": state.get("handoff_summary"),
        "failed_rag_count": state.get("failed_rag_count", 0),
        "consecutive_angry_count": state.get("consecutive_angry_count", 0),
        "max_angry_turns": resolved_settings.agent_max_angry_turns,
        "current_topic": current_topic,
        "current_issue": state.get("current_issue"),
        "perception": perception_dict,
        "intent_draft": _model_to_dict(state.get("intent_draft")),
        "perception_trace": state.get("perception_trace") or {},
        "secondary_intents": perception.secondary_intents if perception else [],
        "clarification": (
            perception.clarification.model_dump() if perception else None
        ),
        "retrieved_docs": [_model_to_dict(doc) for doc in docs],
        "debug_trace": state.get("debug_trace") or {},
    }


def _model_to_dict(value: Any) -> Any:
    if isinstance(value, (PerceptionResult, RetrievedDoc)):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _extract_meta_and_suggestions(
    response: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Replicate frontend meta derivation so JSONL has the same fields."""
    meta: list[str] = []
    perception = response.get("perception") or {}
    if perception.get("intent"):
        meta.append(perception["intent"])
    if perception.get("emotion"):
        meta.append(perception["emotion"])
    if response.get("active_agent"):
        meta.append(response["active_agent"])
    if response.get("dialogue_status") == "awaiting_clarification":
        meta.append("待澄清")
    clarification = response.get("clarification") or {}
    suggestions: list[str] = clarification.get("options") or []
    return meta, suggestions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CGM customer agent demo web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)

    agent = CustomerAgent()
    handler = create_handler(agent)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    assert isinstance(server, ThreadingMixIn)
    LOGGER.info("CGM Agent Demo UI running at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
