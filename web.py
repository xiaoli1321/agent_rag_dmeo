from __future__ import annotations

import argparse
import json
import logging
import pathlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from uuid import uuid4

from .agent.graph import CustomerAgent
from .agent.models import PerceptionResult, RetrievedDoc

LOGGER = logging.getLogger(__name__)

STATIC_DIR = pathlib.Path(__file__).parent / "static"


def _read_index_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


HTML_PAGE = _read_index_html()


def create_handler(agent: CustomerAgent) -> type[BaseHTTPRequestHandler]:
    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "CustomerAgentDemo/0.1"

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self._send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
                return
            if self.path == "/api/health":
                self._send_json({"ok": True})
                return
            # 静态文件
            static_path = STATIC_DIR / self.path.lstrip("/")
            if static_path.is_file() and static_path.parent == STATIC_DIR:
                ext = static_path.suffix
                ctype = {
                    "css": "text/css; charset=utf-8",
                    "js": "application/javascript; charset=utf-8",
                    "html": "text/html; charset=utf-8",
                    "png": "image/png",
                    "svg": "image/svg+xml",
                }.get(ext.lstrip("."), "application/octet-stream")
                self._send_text(
                    static_path.read_text(encoding="utf-8"), content_type=ctype
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                thread_id = str(
                    payload.get("thread_id") or "web-default-thread"
                ).strip()
                if not message:
                    self._send_json(
                        {"error": "message is required"}, status=HTTPStatus.BAD_REQUEST
                    )
                    return
                result = agent.invoke(message, thread_id=thread_id)
                self._send_json(_state_to_response(result, thread_id=thread_id))
            except Exception as exc:  # pragma: no cover - request safety net
                LOGGER.exception("chat request failed")
                self._send_json(
                    {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                )

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
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, *, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoRequestHandler


def _state_to_response(state: dict[str, Any], *, thread_id: str) -> dict[str, Any]:
    perception = state.get("perception")
    docs = state.get("retrieved_docs") or []
    return {
        "thread_id": thread_id,
        "answer": state.get("answer") or "",
        "active_agent": state.get("active_agent"),
        "answer_status": state.get("answer_status"),
        "dialogue_status": state.get("dialogue_status"),
        "handoff_reason": state.get("handoff_reason"),
        "handoff_summary": state.get("handoff_summary"),
        "failed_rag_count": state.get("failed_rag_count", 0),
        "perception": _model_to_dict(perception),
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
