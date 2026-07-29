    let currentAbortController = null;

    async function sendMessage(text) {
      const message = text.trim();
      if (!message) return;
      
      if (currentAbortController) currentAbortController.abort();
      const abortController = new AbortController();
      currentAbortController = abortController;

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
        const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
          signal: abortController.signal,
        });
        
        if (response.ok) {
          await handleStreamResponse(response, message, typingEl);
        } else {
          typingEl.remove();
          addMessage("agent", `请求失败：HTTP ${response.status}`);
        }
      } catch (error) {
        if (error.name === 'AbortError') return;
        typingEl.remove();
        addMessage("agent", `请求失败：${error.message || error}`);
      } finally {
        if (currentAbortController === abortController) currentAbortController = null;
        send.disabled = false;
        input.focus();
      }
    }
    
    async function handleStreamResponse(response, message, typingEl) {
      let fullAnswer = "";
      let finalState = null;
      let streamMsg = null;
      let bubble = null;
      
      if (!response.body) {
        const text = await response.text();
        try {
          const data = JSON.parse(text);
          if (data.answer) {
            typingEl.remove();
            addMessage("agent", data.answer, [], { suggestions: data.clarification?.options || [] });
            if (data.thread_id) {
              threadId = data.thread_id;
              localStorage.setItem("customer_agent_demo_thread_id", threadId);
            }
            if (data.perception || data.retrieved_docs) setState(data);
          }
        } catch {}
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;
      
      while (!done) {
        const readResult = await reader.read();
        done = readResult.done;
        buffer += decoder.decode(readResult.value || new Uint8Array(0), { stream: !done });
        
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        
        for (const part of parts) {
          if (!part.trim()) continue;
          const eventData = parseSSEEvent(part);
          if (!eventData) continue;
          
          const { event, data: dataStr } = eventData;
          let parsed;
          try { parsed = JSON.parse(dataStr); } catch { continue; }
          
          if (event === "answer_token" && parsed.token) {
            fullAnswer += parsed.token;
            if (!streamMsg) {
              typingEl.remove();
              streamMsg = addMessage("agent", "", []);
              bubble = streamMsg.querySelector(".bubble");
            }
            bubble.innerHTML = formatMarkdown(fullAnswer);
            messages.scrollTop = messages.scrollHeight;
          } else if (event === "state") {
            finalState = parsed;
          } else if (event === "error") {
            throw new Error(parsed.error || "流式响应错误");
          }
        }
      }
      
      if (finalState) {
        if (streamMsg) streamMsg.remove();
        
        // Update threadId from server response (catches default-thread mapping)
        threadId = finalState.thread_id || threadId;
        localStorage.setItem("customer_agent_demo_thread_id", threadId);
        
        const meta = [];
        if (finalState.perception?.intent) meta.push(finalState.perception.intent);
        if (finalState.perception?.emotion) meta.push(finalState.perception.emotion);
        if (finalState.active_agent) meta.push(finalState.active_agent);
        if (finalState.dialogue_status === "awaiting_clarification") meta.push("待澄清");
        
        // Use nullish coalescing: only fall back to fullAnswer when answer is null/undefined,
        // NOT when it's an empty string (prevents dropping references when answer is "")
        addMessage("agent", finalState.answer ?? fullAnswer, meta, {
          suggestions: finalState.clarification?.options || [],
        });
        
        setState(finalState);
        
        // Refresh session list, guard against API failure clearing sidebar
        const freshSessions = await fetchSessions();
        if (freshSessions.length > 0 || !sessions.length) {
          sessions = freshSessions;
        }
        renderConversations();
      } else if (streamMsg) {
        // State event never arrived (network interruption, server crash, etc.)
        // Finalize with whatever streaming tokens we accumulated
        streamMsg.remove();
        addMessage("agent", fullAnswer);
      }
    }
    
    function parseSSEEvent(part) {
      const lines = part.split("\n");
      let event = "";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) {
          if (data) data += "\n";
          data += line.slice(6);
        }
      }
      if (!event && !data) return null;
      return { event: event || "message", data };
    }
    

