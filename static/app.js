    const ACTIVE_THREAD_STORAGE_KEY = "customer_agent_demo_thread_id";
    const SESSIONS_STORAGE_KEY = "customer_agent_demo_sessions";

    // Enhanced markdown formatting helper with copy-code snippet support
    function formatMarkdown(text) {
      if (!text) return "";
      let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      
      // Code blocks with syntax box
      html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
        return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
      });
      
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
      
      // Bold
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      
      // Lists
      const lines = html.split('\\n');
      let inList = false;
      const processedLines = [];
      
      for (let line of lines) {
        const listMatch = line.match(/^(\\s*)[-*]\\s+(.+)$/);
        if (listMatch) {
          if (!inList) {
            processedLines.push('<ul class="markdown-list">');
            inList = true;
          }
          processedLines.push(`<li>${listMatch[2]}</li>`);
        } else {
          if (inList) {
            processedLines.push('</ul>');
            inList = false;
          }
          processedLines.push(line);
        }
      }
      if (inList) {
        processedLines.push('</ul>');
      }
      
      return processedLines.join('\\n').replace(/\\n/g, '<br>');
    }

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
    const refs = document.querySelector("#refs");
    const defense = document.querySelector("#defense");
    const thread = document.querySelector("#thread");
    const newThread = document.querySelector("#new-thread");
    const conversations = document.querySelector("#conversations");
    const searchThreads = document.querySelector("#search-threads");
    const toggleSidebar = document.querySelector("#toggle-sidebar");
    const toggleInspector = document.querySelector("#toggle-inspector");
    const closeInspector = document.querySelector("#close-inspector");
    const appLayout = document.querySelector(".app-layout");
    const toggleTheme = document.querySelector("#toggle-theme");
    const themeIcon = document.querySelector("#theme-icon");

    // Tab Navigation Logic inside Inspector
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        tabBtns.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(targetTab)?.classList.add("active");
      });
    });

    // Theme Switcher implementation (Default to light theme)
    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("customer_agent_theme", theme);
      if (theme === "dark") {
        themeIcon.innerHTML = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
      } else {
        themeIcon.innerHTML = `<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>`;
      }
    }

    let currentTheme = localStorage.getItem("customer_agent_theme") || "light";
    applyTheme(currentTheme);

    toggleTheme.addEventListener("click", () => {
      currentTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(currentTheme);
    });

    // Collapse Layout toggles
    toggleSidebar.addEventListener("click", () => {
      appLayout.classList.toggle("sidebar-collapsed");
    });

    toggleInspector.addEventListener("click", () => {
      appLayout.classList.toggle("inspector-collapsed");
    });

    closeInspector.addEventListener("click", () => {
      appLayout.classList.add("inspector-collapsed");
    });

    // Search Conversation Threads
    searchThreads.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      const items = conversations.querySelectorAll(".conversation-item");
      for (const item of items) {
        const title = item.querySelector(".conversation-title").textContent.toLowerCase();
        const id = item.querySelector(".conversation-id").textContent.toLowerCase();
        if (title.includes(q) || id.includes(q)) {
          item.style.display = "flex";
        } else {
          item.style.display = "none";
        }
      }
    });

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
      const empty = messages.querySelector(".empty-state");
      if (empty) empty.remove();
    }

    function resetStatePanel() {
      document.querySelector("#state-intent").textContent = "-";
      document.querySelector("#state-emotion").textContent = "-";
      document.querySelector("#state-agent").textContent = "-";
      document.querySelector("#state-status").textContent = "-";
      document.querySelector("#state-strategy").textContent = "-";
      document.querySelector("#state-reason").textContent = "-";
      document.querySelector("#state-intent").className = "badge";
      document.querySelector("#state-emotion").className = "badge";
      
      refs.innerHTML = '<div class="empty-state-mini">无召回文档数据</div>';
      defense.innerHTML = '<div class="empty-state-mini">无链路追踪数据</div>';
    }

    function deleteSession(idToDelete) {
      if (sessions.length <= 1) {
        alert("请保留至少一个会话。");
        return;
      }
      const index = sessions.findIndex(s => s.threadId === idToDelete);
      if (index === -1) return;
      
      sessions.splice(index, 1);
      if (threadId === idToDelete) {
        threadId = sessions[0].threadId;
      }
      saveSessions();
      renderSession();
    }

    function renderConversations() {
      conversations.innerHTML = "";
      for (const session of sessions) {
        const item = document.createElement("div");
        item.className = `conversation-item${session.threadId === threadId ? " active" : ""}`;
        
        const content = document.createElement("div");
        content.className = "conversation-content";
        
        const title = document.createElement("span");
        title.className = "conversation-title";
        title.textContent = session.title || "新对话";
        
        const id = document.createElement("small");
        id.className = "conversation-id";
        id.textContent = session.threadId;
        
        content.appendChild(title);
        content.appendChild(id);
        item.appendChild(content);
        
        // Delete button
        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-delete-thread";
        deleteBtn.type = "button";
        deleteBtn.title = "删除对话";
        deleteBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
        
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteSession(session.threadId);
        });
        
        item.appendChild(deleteBtn);
        item.addEventListener("click", () => switchSession(session.threadId));
        conversations.appendChild(item);
      }
    }

    function renderMessages() {
      const session = activeSession();
      messages.innerHTML = "";
      if (!session || !session.messages.length) {
        messages.innerHTML = `
          <div class="empty-state">
            <div class="hero-badge">
              <span class="pulse-dot"></span>
              <span>LangGraph Multi-Agent 架构</span>
            </div>
            <h2 class="empty-state-title">CGM 智能血糖客服</h2>
            <p class="empty-state-subtitle">内置 Self-RAG/CRAG 双重防护网与 Multi-Agent 分流协同架构，保障医疗级客服的高准确度与极低幻觉率。</p>

            <div class="hero-features-grid">
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                  <span>Self-RAG 证据防护网</span>
                </div>
                <div class="hero-feature-desc">通过 LLM Grader 逐级判定文档真实性与关联度，防范回答幻觉与跨文本捏造。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                  <span>Swarm Agent 动态编排</span>
                </div>
                <div class="hero-feature-desc">针对产品咨询、订单售后及负面情绪，实现秒级 Swarm 路由与安抚。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                  <span>Qdrant 混合向量检索</span>
                </div>
                <div class="hero-feature-desc">高维 Dense 向量与关键词 BM25 稀疏检索 RRF 融合，准确召回产品说明。</div>
              </div>
              <div class="hero-feature-card">
                <div class="hero-feature-title">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--accent-solid)" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                  <span>全链路可视化 Trace</span>
                </div>
                <div class="hero-feature-desc">调试控制台实时呈现意图解析、召回得分、C1 防线闭环及状态跳跃。</div>
              </div>
            </div>
          </div>
        `;
        return;
      }
      for (const message of session.messages) {
        addMessage(message.role, message.text, message.meta || [], {
          persist: false,
          suggestions: message.suggestions || [],
        });
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
      
      // Avatar Graphic (SVG)
      const avatarEl = document.createElement("div");
      avatarEl.className = "avatar";
      if (role === "user") {
        avatarEl.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
      } else {
        avatarEl.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none"><path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zM8 11h8M12 11v6m-4 4h8"></path><rect x="4" y="8" width="16" height="10" rx="2"></rect></svg>`;
      }
      item.appendChild(avatarEl);
      
      const body = document.createElement("div");
      body.className = "message-body";
      
      const bubbleWrapper = document.createElement("div");
      bubbleWrapper.className = "bubble-wrapper";
      
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.innerHTML = formatMarkdown(text);
      bubbleWrapper.appendChild(bubble);
      body.appendChild(bubbleWrapper);

      // Message Action Toolbar (Copy button)
      if (role === "agent" && text) {
        const actionsBar = document.createElement("div");
        actionsBar.className = "message-actions-bar";

        const copyBtn = document.createElement("button");
        copyBtn.className = "btn-msg-action";
        copyBtn.type = "button";
        copyBtn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg><span>复制</span>`;
        copyBtn.addEventListener("click", () => {
          navigator.clipboard.writeText(text);
          copyBtn.querySelector("span").textContent = "已复制";
          setTimeout(() => copyBtn.querySelector("span").textContent = "复制", 2000);
        });
        actionsBar.appendChild(copyBtn);
        body.appendChild(actionsBar);
      }

      if (meta.length) {
        const metaEl = document.createElement("div");
        metaEl.className = "message-meta";
        for (const label of meta) {
          const pill = document.createElement("span");
          pill.className = "badge";
          if (label === "human_handoff" || label === "complain" || label === "negative") {
            pill.classList.add("badge-danger");
          } else if (label.includes("FAQ") || label.includes("faq") || label.includes("consultation")) {
            pill.classList.add("badge-accent");
          } else if (label === "positive") {
            pill.classList.add("badge-accent");
          } else {
            pill.classList.add("badge-info");
          }
          pill.textContent = label;
          metaEl.appendChild(pill);
        }
        body.appendChild(metaEl);
      }

      const suggestions = options.suggestions || [];
      if (role === "agent" && suggestions.length) {
        const suggestionsEl = document.createElement("div");
        suggestionsEl.className = "clarification-options";
        for (const suggestion of suggestions) {
          const optionButton = document.createElement("button");
          optionButton.type = "button";
          optionButton.className = "clarification-option";
          optionButton.textContent = suggestion;
          optionButton.addEventListener("click", () => sendMessage(suggestion));
          suggestionsEl.appendChild(optionButton);
        }
        body.appendChild(suggestionsEl);
      }
      
      item.appendChild(body);
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      
      if (options.persist !== false) {
        const session = activeSession();
        if (session) {
          session.messages.push({ role, text, meta, suggestions });
          saveSessions();
          renderConversations();
        }
      }
    }

    function setState(payload, options = {}) {
      const perception = payload.perception || {};
      
      const intentBadge = document.querySelector("#state-intent");
      intentBadge.textContent = perception.intent || "-";
      intentBadge.className = "badge";
      if (perception.intent) {
        if (perception.intent.includes("human_handoff") || perception.intent.includes("complain")) {
          intentBadge.classList.add("badge-danger");
        } else if (perception.intent.includes("faq")) {
          intentBadge.classList.add("badge-accent");
        } else {
          intentBadge.classList.add("badge-info");
        }
      }
      
      const emotionBadge = document.querySelector("#state-emotion");
      emotionBadge.textContent = perception.emotion || "-";
      emotionBadge.className = "badge";
      if (perception.emotion) {
        if (perception.emotion === "angry" || perception.emotion === "anxious") {
          emotionBadge.classList.add("badge-danger");
        } else if (perception.emotion === "happy" || perception.emotion === "satisfied") {
          emotionBadge.classList.add("badge-accent");
        } else {
          emotionBadge.classList.add("badge-info");
        }
      }

      document.querySelector("#state-agent").textContent = payload.active_agent || "-";
      document.querySelector("#state-status").textContent = payload.dialogue_status || payload.answer_status || "-";
      document.querySelector("#state-secondary-intents").textContent = (perception.secondary_intents || []).join("、") || "-";
      const clarification = perception.clarification || {};
      document.querySelector("#state-clarification").textContent = clarification.needed
        ? `${clarification.reason || "信息不足"}：${(clarification.missing_slots || []).join("、")}`
        : "-";
      document.querySelector("#state-strategy").textContent = payload.debug_trace?.retrieval_strategy || "-";
      const decision = payload.perception_trace?.policy_decision || {};
      document.querySelector("#state-reason").textContent = decision.policy_reason || payload.debug_trace?.evidence_reason || "-";
      
      refs.innerHTML = "";
      const docs = payload.retrieved_docs || [];
      if (!docs.length) {
        refs.innerHTML = '<div class="empty-state-mini">无召回文档数据</div>';
      } else {
        for (const doc of docs) {
          const card = document.createElement("div");
          card.className = "ref-card";
          
          const header = document.createElement("div");
          header.className = "ref-header";
          
          const title = document.createElement("span");
          title.className = "ref-title";
          title.textContent = doc.source_title;
          title.title = doc.source_title;
          
          const score = document.createElement("span");
          score.className = "ref-score";
          const isRrfRank = doc.retrieval_source === "hybrid" && doc.retrieval_rank;
          const scoreVal = Number(doc.score || 0);
          score.textContent = isRrfRank ? `RRF #${doc.retrieval_rank}` : scoreVal.toFixed(3);
          
          header.appendChild(title);
          header.appendChild(score);
          card.appendChild(header);
          
          const meta = document.createElement("div");
          meta.className = "ref-meta";
          meta.textContent = doc.retrieval_source || 'retrieved';
          card.appendChild(meta);
          
          if (!isRrfRank) {
            const scoreBar = document.createElement("div");
            scoreBar.className = "score-bar";
            const scoreFill = document.createElement("div");
            scoreFill.className = "score-fill";
            scoreFill.style.width = `${Math.min(100, scoreVal * 100)}%`;
            scoreBar.appendChild(scoreFill);
            card.appendChild(scoreBar);
          }
          
          card.style.cursor = "pointer";
          card.addEventListener("click", () => {
            document.querySelector("#modal-source").textContent = `${doc.source_title} (score: ${(Number(doc.score) || 0).toFixed(3)})`;
            document.querySelector("#modal-body").textContent = doc.chunk_text || "(空)";
            document.querySelector("#doc-modal").style.display = "";
          });
          refs.appendChild(card);
        }
      }

      defense.innerHTML = "";
      const steps = payload.debug_trace?.pipeline_steps || [];
      const grades = payload.debug_trace?.document_grades || [];
      const filteredGrades = grades.filter((item) => item.binary_score === "no").slice(0, 3);
      
      if (!steps.length && !filteredGrades.length) {
        defense.innerHTML = '<div class="empty-state-mini">无链路追踪数据</div>';
      } else {
        for (const step of steps) {
          const tStep = document.createElement("div");
          tStep.className = "timeline-step";
          
          const node = document.createElement("div");
          node.className = "step-node";
          tStep.appendChild(node);
          
          const card = document.createElement("div");
          card.className = "step-card";
          
          const header = document.createElement("div");
          header.className = "step-card-header";
          
          const name = document.createElement("span");
          name.className = "step-name";
          name.textContent = step.name || "-";
          
          const status = document.createElement("span");
          status.className = "step-status";
          status.textContent = step.status || "-";
          
          header.appendChild(name);
          header.appendChild(status);
          card.appendChild(header);
          
          const desc = document.createElement("div");
          desc.className = "step-desc";
          const summary = step.output_summary || "";
          const blocked = step.blocked_reason ? ` · ${step.blocked_reason}` : "";
          desc.textContent = `${summary}${blocked}`;
          card.appendChild(desc);
          
          tStep.appendChild(card);
          
          if (step.status === "passed" || step.status === "completed" || step.status === "success") {
            tStep.classList.add("success");
          } else if (step.status === "blocked" || step.status === "failed") {
            tStep.classList.add("failed");
          } else if (step.status === "running") {
            tStep.classList.add("running");
          }
          
          defense.appendChild(tStep);
        }
        
        for (const grade of filteredGrades) {
          const tStep = document.createElement("div");
          tStep.className = "timeline-step failed";
          
          const node = document.createElement("div");
          node.className = "step-node";
          tStep.appendChild(node);
          
          const card = document.createElement("div");
          card.className = "step-card";
          
          const header = document.createElement("div");
          header.className = "step-card-header";
          
          const name = document.createElement("span");
          name.className = "step-name";
          name.textContent = `grader 拦截 (${grade.grader || "unknown"})`;
          
          const status = document.createElement("span");
          status.className = "step-status";
          status.textContent = grade.failure_type || "unknown";
          
          header.appendChild(name);
          header.appendChild(status);
          card.appendChild(header);
          
          const desc = document.createElement("div");
          desc.className = "step-desc";
          desc.innerHTML = `文档 <strong>${grade.source_title}</strong> 未通过校验：${grade.reason} (第 ${Number(grade.attempt || 0) + 1} 次尝试)`;
          card.appendChild(desc);
          
          tStep.appendChild(card);
          defense.appendChild(tStep);
        }
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
      input.style.height = "auto";
      send.disabled = true;
      
      // Add typing indicator element
      const typingEl = document.createElement("div");
      typingEl.className = "typing-indicator-wrapper";
      typingEl.innerHTML = `
        <div class="typing-bubble">
          <span></span>
          <span></span>
          <span></span>
        </div>
      `;
      messages.appendChild(typingEl);
      messages.scrollTop = messages.scrollHeight;
      
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
        });
        const data = await response.json();
        
        // Remove typing indicator
        typingEl.remove();
        
        if (!response.ok) throw new Error(data.error || "请求失败");
        const meta = [];
        if (data.perception?.intent) meta.push(data.perception.intent);
        if (data.perception?.emotion) meta.push(data.perception.emotion);
        if (data.active_agent) meta.push(data.active_agent);
        if (data.dialogue_status === "awaiting_clarification") meta.push("待澄清");
        
        addMessage("agent", data.answer || "", meta, {
          suggestions: data.clarification?.options || [],
        });
        const session = activeSession();
        if (session && session.title === "新对话") {
          session.title = message.slice(0, 18);
        }
        setState(data);
        saveSessions();
        renderConversations();
      } catch (error) {
        typingEl.remove();
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

    // Auto grow input height dynamically
    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = `${input.scrollHeight}px`;
    });

    // Event listener delegation for sample prompt buttons
    document.addEventListener("click", (e) => {
      const sampleBtn = e.target.closest(".sample-btn");
      if (sampleBtn) {
        const text = sampleBtn.querySelector("span")?.textContent || sampleBtn.textContent;
        sendMessage(text);
      }
    });

    newThread.addEventListener("click", () => {
      const session = createSession();
      sessions = [session, ...sessions];
      threadId = session.threadId;
      saveSessions();
      renderSession();
      input.focus();
    });
    renderSession();

    document.querySelector("#modal-close-btn").addEventListener("click", () => {
      document.querySelector("#doc-modal").style.display = "none";
    });
    document.querySelector("#doc-modal").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) {
        document.querySelector("#doc-modal").style.display = "none";
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        document.querySelector("#doc-modal").style.display = "none";
      }
    });

