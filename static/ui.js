    function renderThreadId() {
      thread.textContent = `thread_id=${threadId}`;
    }

    function activeSession() {
      return sessions.find((s) => s.thread_id === threadId);
    }

    function clearEmpty() {
      const empty = messages.querySelector(".empty-state");
      if (empty) empty.remove();
    }

    function resetStatePanel() {
      document.querySelector("#state-entities").textContent = "-";
      document.querySelector("#state-missing-slots").textContent = "-";
      document.querySelector("#state-secondary-intents").textContent = "-";
      document.querySelector("#state-clarification").textContent = "-";
      document.querySelector("#state-strategy").textContent = "-";
      document.querySelector("#state-reason").textContent = "-";
      document.querySelector("#state-intent").className = "badge";
      document.querySelector("#state-emotion").className = "badge";
      document.querySelector("#state-angry-count").textContent = "0/2 次";
      document.querySelector("#state-angry-count").className = "badge";
      
      refs.innerHTML = '<div class="empty-state-mini">无召回文档数据</div>';
      defense.innerHTML = '<div class="empty-state-mini">无链路追踪数据</div>';
    }

    async function deleteSession(idToDelete) {
      const resp = await fetch(`/api/conversations/${encodeURIComponent(idToDelete)}`, { method: "DELETE" });
      if (!resp.ok) return;
      sessions = sessions.filter(s => s.thread_id !== idToDelete);
      if (threadId === idToDelete) {
        threadId = sessions.length > 0 ? sessions[0].thread_id : "";
        localStorage.setItem("customer_agent_demo_thread_id", threadId);
      }
      await renderSession();
    }

    function renderConversations() {
      conversations.innerHTML = "";
      for (const session of sessions) {
        const item = document.createElement("div");
        item.className = `conversation-item${session.thread_id === threadId ? " active" : ""}`;
        
        const content = document.createElement("div");
        content.className = "conversation-content";
        
        const title = document.createElement("span");
        title.className = "conversation-title";
        title.textContent = session.title || "新对话";
        
        const id = document.createElement("small");
        id.className = "conversation-id";
        id.textContent = session.thread_id;
        
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
          deleteSession(session.thread_id);
        });
        
        item.appendChild(deleteBtn);
        item.addEventListener("click", () => switchSession(session.thread_id));
        conversations.appendChild(item);
      }
    }

    async function renderMessages() {
      const session = activeSession();
      messages.innerHTML = "";
      if (!session) {
        messages.innerHTML = `
          <div class="empty-state">
            <div class="hero-badge">
              <span class="pulse-dot"></span>
              <span>LangGraph Multi-Agent 架构</span>
            </div>
            <h2 class="empty-state-title">CGM 智能血糖客服</h2>
            <p class="empty-state-subtitle">内置 Self-RAG/CRAG 双重防护网与 Multi-Agent 分流协同架构，保障医疗级客服的高准确度与极低幻觉率。</p>
          </div>
        `;
        return;
      }
      let msgs;
      if (session._legacy) {
        // Transitional: use messages stored in localStorage for legacy sessions
        msgs = session._messages || [];
        if (session._state) {
          // Inject state into last agent message for inspector panel
          for (let i = msgs.length - 1; i >= 0; i--) {
            if (msgs[i].role === "agent") {
              msgs[i].state = session._state;
              break;
            }
          }
        }
      } else {
        msgs = await fetchSessionMessages(session.thread_id);
      }
      if (!msgs || !msgs.length) {
        messages.innerHTML = `
          <div class="empty-state">
            <div class="hero-badge">
              <span class="pulse-dot"></span>
              <span>LangGraph Multi-Agent 架构</span>
            </div>
            <h2 class="empty-state-title">CGM 智能血糖客服</h2>
            <p class="empty-state-subtitle">内置 Self-RAG/CRAG 双重防护网与 Multi-Agent 分流协同架构，保障医疗级客服的高准确度与极低幻觉率。</p>
          </div>
        `;
        return;
      }
      let lastState = null;
      for (const msg of msgs) {
        addMessage(msg.role, msg.text, msg.meta || [], {
          suggestions: msg.suggestions || [],
        });
        if (msg.role === "agent" && msg.state) {
          lastState = msg.state;
        }
      }

      // For legacy sessions, always try to restore state even if injection missed
      if (!lastState && session._legacy && session._state) {
        lastState = session._state;
      }

      if (lastState) {
        setState(lastState);
      } else {
        resetStatePanel();
      }
    }

    async function renderSession() {
      renderThreadId();
      renderConversations();
      await renderMessages();
    }

    async function switchSession(nextThreadId) {
      threadId = nextThreadId;
      localStorage.setItem("customer_agent_demo_thread_id", threadId);
      await renderSession();
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
      return item;
    }

    function setState(payload) {
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
        if (perception.emotion === "angry" || perception.emotion === "anxious" || perception.emotion === "愤怒" || perception.emotion === "不满") {
          emotionBadge.classList.add("badge-danger");
        } else if (perception.emotion === "happy" || perception.emotion === "satisfied" || perception.emotion === "平静") {
          emotionBadge.classList.add("badge-accent");
        } else {
          emotionBadge.classList.add("badge-info");
        }
      }

      const angryCountBadge = document.querySelector("#state-angry-count");
      const angryCount = payload.consecutive_angry_count || 0;
      const maxAngryTurns = payload.max_angry_turns || 3;
      angryCountBadge.textContent = `${angryCount}/${maxAngryTurns} 次`;
      angryCountBadge.className = "badge";
      if (angryCount >= maxAngryTurns) {
        angryCountBadge.classList.add("badge-danger");
        angryCountBadge.textContent = `${angryCount}/${maxAngryTurns} 次 (触发转人工)`;
      } else if (angryCount > 0) {
        angryCountBadge.classList.add("badge-danger");
      }

      document.querySelector("#state-agent").textContent = payload.active_agent || "-";
      document.querySelector("#state-status").textContent = payload.dialogue_status || payload.answer_status || "-";

      // 槽位实体 (Entities) 具体高亮渲染
      const entities = perception.entities || {};
      const entitiesContainer = document.querySelector("#state-entities");
      entitiesContainer.innerHTML = "";
      const productVal = entities.product || payload.current_topic;
      const entityItems = [
        { key: "product", label: "产品型号", value: productVal, color: "badge-accent" },
        { key: "issue", label: "故障/现象", value: entities.issue, color: "badge-danger" },
        { key: "requested_action", label: "业务动作", value: entities.requested_action, color: "badge-info" }
      ].filter(item => item.value);

      if (entityItems.length) {
        entityItems.forEach(item => {
          const pill = document.createElement("span");
          pill.className = `badge ${item.color}`;
          pill.style.marginRight = "6px";
          pill.style.marginBottom = "4px";
          pill.style.display = "inline-block";
          const strong = document.createElement('strong');
          strong.textContent = item.label + ': ';
          pill.textContent = '';
          pill.appendChild(strong);
          pill.append(item.value);
          entitiesContainer.appendChild(pill);
        });
      } else {
        entitiesContainer.textContent = "未提取到明确实体";
      }

      // 缺失槽位 (Missing Slots) 具体高亮渲染
      const clarification = perception.clarification || {};
      const missingContainer = document.querySelector("#state-missing-slots");
      missingContainer.innerHTML = "";
      const missingSlots = clarification.missing_slots || [];
      if (missingSlots.length) {
        missingSlots.forEach(slot => {
          const pill = document.createElement("span");
          pill.className = "badge badge-danger";
          pill.style.marginRight = "6px";
          pill.style.display = "inline-block";
          pill.textContent = `⚠️ 缺失: ${slot}`;
          missingContainer.appendChild(pill);
        });
      } else {
        missingContainer.innerHTML = '<span class="badge badge-accent">✓ 槽位健全 (无缺失)</span>';
      }

      document.querySelector("#state-secondary-intents").textContent = (perception.secondary_intents || []).join("、") || "-";
      document.querySelector("#state-clarification").textContent = clarification.needed
        ? `${clarification.reason || "缺少关键槽位"}：${(clarification.missing_slots || []).join("、")}`
        : "无需澄清";
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
          const isRrfRank = doc.retrieval_source === "hybrid" && doc.retrieval_rank != null;
          const scoreVal = Math.max(0, Number(doc.score) || 0);
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
          const strong = document.createElement('strong');
          strong.textContent = grade.source_title || '';
          desc.textContent = '';
          desc.appendChild(document.createTextNode('文档 '));
          desc.appendChild(strong);
          desc.append(` 未通过校验：${grade.reason} (第 ${Number(grade.attempt || 0) + 1} 次尝试)`);
          card.appendChild(desc);
          
          tStep.appendChild(card);
          defense.appendChild(tStep);
        }
      }
    }
