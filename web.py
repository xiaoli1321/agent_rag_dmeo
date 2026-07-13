from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from uuid import uuid4

from customer_agent_demo.agent.graph import CustomerAgent, new_thread_id
from customer_agent_demo.agent.models import PerceptionResult, RetrievedDoc


LOGGER = logging.getLogger(__name__)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CGM 智能客服 Agent Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --agent: #eef7f5;
      --user: #17202a;
      --warn: #9a3412;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    .header-inner {
      max-width: 1080px;
      margin: 0 auto;
      padding: 18px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #16a34a;
      box-shadow: 0 0 0 3px rgba(22, 163, 74, .12);
    }

    main {
      max-width: 1080px;
      width: 100%;
      margin: 0 auto;
      padding: 18px 20px 12px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 16px;
      min-height: 0;
    }

    .chat {
      min-height: 0;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      display: grid;
      grid-template-rows: 1fr auto;
      overflow: hidden;
    }

    .messages {
      min-height: 420px;
      max-height: calc(100vh - 205px);
      overflow-y: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      max-width: 78%;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .message.user {
      align-self: flex-end;
      align-items: flex-end;
    }

    .bubble {
      border-radius: 8px;
      padding: 12px 13px;
      line-height: 1.58;
      font-size: 15px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .agent .bubble {
      background: var(--agent);
      border: 1px solid #c9e7e1;
    }

    .user .bubble {
      background: var(--user);
      color: #fff;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      background: #fff;
    }

    .composer {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      background: #fbfcfd;
    }

    textarea {
      width: 100%;
      min-height: 46px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      line-height: 1.45;
      color: var(--ink);
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .12);
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      min-width: 82px;
      font: inherit;
      font-weight: 650;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }

    button:hover { background: var(--accent-strong); }
    button:disabled { cursor: wait; opacity: .65; }

    aside {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
    }

    .panel h2 {
      margin: 0 0 10px;
      font-size: 14px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .samples {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .sample,
    .conversation {
      width: 100%;
      min-width: 0;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      text-align: left;
      font-size: 13px;
      font-weight: 500;
    }

    .sample:hover,
    .conversation:hover {
      border-color: var(--accent);
      background: #f0fdfa;
      color: var(--accent-strong);
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }

    .panel-head h2 {
      margin: 0;
    }

    .mini-button {
      min-width: 0;
      height: 28px;
      padding: 0 9px;
      font-size: 12px;
      font-weight: 650;
    }

    .conversations {
      display: grid;
      gap: 8px;
      max-height: 180px;
      overflow-y: auto;
    }

    .conversation {
      display: grid;
      gap: 3px;
    }

    .conversation.active {
      border-color: var(--accent);
      background: #f0fdfa;
      color: var(--accent-strong);
    }

    .conversation small {
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .kv {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }

    .kv strong {
      display: block;
      color: var(--ink);
      font-size: 13px;
      margin-bottom: 2px;
    }

    .refs {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }

    .ref {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }

    .defense {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .step {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      font-size: 12px;
      line-height: 1.45;
      color: var(--muted);
    }

    .step strong {
      color: var(--ink);
      font-size: 12px;
      margin: 0 0 2px;
    }

    .failures {
      display: grid;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }

    .empty {
      margin: auto;
      text-align: center;
      color: var(--muted);
      max-width: 360px;
      line-height: 1.6;
    }

    footer {
      max-width: 1080px;
      width: 100%;
      margin: 0 auto;
      padding: 0 20px 14px;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .ghost-button {
      min-width: 72px;
      height: 30px;
      padding: 0 10px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-size: 12px;
      font-weight: 600;
    }

    .ghost-button:hover {
      border-color: var(--accent);
      background: #f0fdfa;
      color: var(--accent-strong);
    }

    @media (max-width: 860px) {
      .header-inner { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      aside { order: -1; }
      .messages { max-height: none; min-height: 360px; }
      .message { max-width: 92%; }
      .composer { grid-template-columns: 1fr; }
      button { height: 44px; }
      footer { align-items: flex-start; flex-direction: column; }
      .ghost-button { height: 34px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="header-inner">
        <h1>CGM 智能客服 Agent Demo</h1>
        <div class="status"><span class="dot"></span><span id="mode">运行中</span></div>
      </div>
    </header>
    <main>
      <section class="chat" aria-label="chat">
        <div id="messages" class="messages">
          <div class="empty">可以直接开始对话。当前页面调用 customer_agent_demo 的 LangGraph Agent。</div>
        </div>
        <form id="form" class="composer">
          <textarea id="input" name="message" placeholder="输入你的问题" autocomplete="off"></textarea>
          <button id="send" type="submit">发送</button>
        </form>
      </section>
      <aside>
        <section class="panel">
          <div class="panel-head">
            <h2>对话</h2>
            <button id="new-thread" class="mini-button" type="button">新增对话</button>
          </div>
          <div id="conversations" class="conversations"></div>
        </section>
        <section class="panel">
          <h2>示例</h2>
          <div class="samples">
            <button class="sample" type="button">Dexcom G7 可以戴着洗澡吗？</button>
            <button class="sample" type="button">连接码是几位数？</button>
            <button class="sample" type="button">我的订单为什么还没发货？</button>
            <button class="sample" type="button">你们这个传感器太差了，刚贴上就坏了，我要投诉，马上给我人工！</button>
          </div>
        </section>
        <section class="panel">
          <h2>本轮状态</h2>
          <div id="state" class="kv">
            <div><strong>意图</strong><span>-</span></div>
            <div><strong>情绪</strong><span>-</span></div>
            <div><strong>Agent</strong><span>-</span></div>
            <div><strong>回答状态</strong><span>-</span></div>
          </div>
          <div id="refs" class="refs"></div>
        </section>
        <section class="panel">
          <h2>C1 防线</h2>
          <div id="defense" class="defense"></div>
        </section>
        <section class="panel">
          <h2>失败类型</h2>
          <div class="failures">
            <div><strong>knowledge_missing</strong>：补知识源后重新入库</div>
            <div><strong>retrieval_mismatch</strong>：调 chunk、hybrid/rerank 或 query rewrite</div>
            <div><strong>hallucination</strong>：收紧 prompt，生成后校验并拒答</div>
            <div><strong>format_unstable</strong>：结构化输出或后处理引用格式</div>
          </div>
        </section>
      </aside>
    </main>
    <footer>
      <span id="thread"></span>
    </footer>
  </div>
  <script>
    const ACTIVE_THREAD_STORAGE_KEY = "customer_agent_demo_thread_id";
    const SESSIONS_STORAGE_KEY = "customer_agent_demo_sessions";
    function createThreadId() {
      const random = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2);
      return `web-${random}`;
    }
    function createSession() {
      return {
        threadId: createThreadId(),
        title: "新对话",
        messages: [],
        state: null,
        createdAt: Date.now(),
      };
    }
    function loadSessions() {
      try {
        const parsed = JSON.parse(localStorage.getItem(SESSIONS_STORAGE_KEY) || "[]");
        if (Array.isArray(parsed) && parsed.length) return parsed;
      } catch (_) {}
      const first = createSession();
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify([first]));
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, first.threadId);
      return [first];
    }
    let sessions = loadSessions();
    let threadId = localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY) || sessions[0].threadId;
    if (!sessions.some((session) => session.threadId === threadId)) {
      threadId = sessions[0].threadId;
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, threadId);
    }
    const messages = document.querySelector("#messages");
    const form = document.querySelector("#form");
    const input = document.querySelector("#input");
    const send = document.querySelector("#send");
    const state = document.querySelector("#state");
    const refs = document.querySelector("#refs");
    const defense = document.querySelector("#defense");
    const thread = document.querySelector("#thread");
    const newThread = document.querySelector("#new-thread");
    const conversations = document.querySelector("#conversations");

    function renderThreadId() {
      thread.textContent = `thread_id=${threadId}`;
    }

    function activeSession() {
      return sessions.find((session) => session.threadId === threadId);
    }

    function saveSessions() {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
      localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, threadId);
    }

    function clearEmpty() {
      const empty = messages.querySelector(".empty");
      if (empty) empty.remove();
    }

    function resetStatePanel() {
      state.innerHTML = `
        <div><strong>意图</strong><span>-</span></div>
        <div><strong>情绪</strong><span>-</span></div>
        <div><strong>Agent</strong><span>-</span></div>
        <div><strong>回答状态</strong><span>-</span></div>
      `;
      refs.innerHTML = "";
      defense.innerHTML = "";
    }

    function renderConversations() {
      conversations.innerHTML = "";
      for (const session of sessions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `conversation${session.threadId === threadId ? " active" : ""}`;
        const label = document.createElement("span");
        label.textContent = session.title || "新对话";
        const id = document.createElement("small");
        id.textContent = session.threadId;
        button.appendChild(label);
        button.appendChild(id);
        button.addEventListener("click", () => switchSession(session.threadId));
        conversations.appendChild(button);
      }
    }

    function renderMessages() {
      const session = activeSession();
      messages.innerHTML = "";
      if (!session || !session.messages.length) {
        messages.innerHTML = '<div class="empty">可以直接开始对话。当前页面调用 customer_agent_demo 的 LangGraph Agent。</div>';
        return;
      }
      for (const message of session.messages) {
        addMessage(message.role, message.text, message.meta || [], { persist: false });
      }
    }

    function renderSession() {
      renderThreadId();
      renderConversations();
      renderMessages();
      const session = activeSession();
      if (session?.state) {
        setState(session.state, { persist: false });
      } else {
        resetStatePanel();
      }
    }

    function switchSession(nextThreadId) {
      threadId = nextThreadId;
      saveSessions();
      renderSession();
      input.focus();
    }

    function addMessage(role, text, meta = [], options = {}) {
      clearEmpty();
      const item = document.createElement("div");
      item.className = `message ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      item.appendChild(bubble);
      if (meta.length) {
        const metaEl = document.createElement("div");
        metaEl.className = "meta";
        for (const label of meta) {
          const pill = document.createElement("span");
          pill.className = "pill";
          pill.textContent = label;
          metaEl.appendChild(pill);
        }
        item.appendChild(metaEl);
      }
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.messages.push({ role, text, meta });
          saveSessions();
          renderConversations();
        }
      }
    }

    function setState(payload, options = {}) {
      const perception = payload.perception || {};
      state.innerHTML = `
        <div><strong>意图</strong><span>${perception.intent || "-"}</span></div>
        <div><strong>情绪</strong><span>${perception.emotion || "-"}</span></div>
        <div><strong>Agent</strong><span>${payload.active_agent || "-"}</span></div>
        <div><strong>回答状态</strong><span>${payload.answer_status || "-"}</span></div>
        <div><strong>检索策略</strong><span>${payload.debug_trace?.retrieval_strategy || "-"}</span></div>
        <div><strong>证据原因</strong><span>${payload.debug_trace?.evidence_reason || "-"}</span></div>
      `;
      refs.innerHTML = "";
      for (const doc of payload.retrieved_docs || []) {
        const ref = document.createElement("div");
        ref.className = "ref";
        ref.textContent = `${doc.source_title} · chunk #${doc.chunk_index} · score ${Number(doc.score).toFixed(3)}`;
        refs.appendChild(ref);
      }
      defense.innerHTML = "";
      const steps = payload.debug_trace?.pipeline_steps || [];
      for (const step of steps) {
        const item = document.createElement("div");
        item.className = "step";
        const title = document.createElement("strong");
        title.textContent = `${step.name || "-"} · ${step.status || "-"}`;
        const body = document.createElement("div");
        const summary = step.output_summary || "";
        const blocked = step.blocked_reason ? ` · ${step.blocked_reason}` : "";
        body.textContent = `${summary}${blocked}`;
        item.appendChild(title);
        item.appendChild(body);
        defense.appendChild(item);
      }
      const grades = payload.debug_trace?.document_grades || [];
      for (const grade of grades.filter((item) => item.binary_score === "no").slice(0, 3)) {
        const item = document.createElement("div");
        item.className = "step";
        const title = document.createElement("strong");
        title.textContent = `grader 拦截 · ${grade.failure_type || "unknown"} · ${grade.grader || "unknown"} · 第 ${Number(grade.attempt || 0) + 1} 次`;
        const body = document.createElement("div");
        body.textContent = `${grade.source_title} chunk #${grade.chunk_index} · ${grade.reason}`;
        item.appendChild(title);
        item.appendChild(body);
        defense.appendChild(item);
      }
      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.state = payload;
          saveSessions();
        }
      }
    }

    async function sendMessage(text) {
      const message = text.trim();
      if (!message) return;
      addMessage("user", message);
      input.value = "";
      send.disabled = true;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "请求失败");
        const meta = [];
        if (data.perception?.intent) meta.push(data.perception.intent);
        if (data.perception?.emotion) meta.push(data.perception.emotion);
        if (data.active_agent) meta.push(data.active_agent);
        addMessage("agent", data.answer || "", meta);
        const session = activeSession();
        if (session && session.title === "新对话") {
          session.title = message.slice(0, 18);
        }
        setState(data);
        saveSessions();
        renderConversations();
      } catch (error) {
        addMessage("agent", `请求失败：${error.message || error}`);
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage(input.value);
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    for (const button of document.querySelectorAll(".sample")) {
      button.addEventListener("click", () => sendMessage(button.textContent));
    }

    newThread.addEventListener("click", () => {
      const session = createSession();
      sessions = [session, ...sessions];
      threadId = session.threadId;
      saveSessions();
      renderSession();
      input.focus();
    });

    renderSession();
  </script>
</body>
</html>
"""


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
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                thread_id = str(payload.get("thread_id") or "web-default-thread").strip()
                if not message:
                    self._send_json({"error": "message is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                result = agent.invoke(message, thread_id=thread_id)
                self._send_json(_state_to_response(result, thread_id=thread_id))
            except Exception as exc:  # pragma: no cover - request safety net
                LOGGER.exception("chat request failed")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
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
        "handoff_reason": state.get("handoff_reason"),
        "handoff_summary": state.get("handoff_summary"),
        "failed_rag_count": state.get("failed_rag_count", 0),
        "perception": _model_to_dict(perception),
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
    parser = argparse.ArgumentParser(description="Run the CGM customer agent demo web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
