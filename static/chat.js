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
      
      // Create agent message element (with avatar) immediately,
      // show typing dots inside the bubble while waiting for stream
      const streamMsg = addMessage("agent", "", []);
      const bubble = streamMsg.querySelector(".bubble");
      bubble.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;
      messages.scrollTop = messages.scrollHeight;
      
      try {
        const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, thread_id: threadId }),
          signal: abortController.signal,
        });
        
        if (response.ok) {
          await handleStreamResponse(response, message, streamMsg, bubble);
        } else {
          bubble.innerHTML = formatMarkdown(`请求失败：HTTP ${response.status}`);
        }
      } catch (error) {
        if (error.name === 'AbortError') return;
        bubble.innerHTML = formatMarkdown(`请求失败：${error.message || error}`);
      } finally {
        if (currentAbortController === abortController) currentAbortController = null;
        send.disabled = false;
        input.focus();
      }
    }
    
    async function handleStreamResponse(response, message, streamMsg, bubble) {
      // response.body 为空时优雅降级
      if (!response.body) {
        const text = await response.text();
        try {
          const data = JSON.parse(text);
          if (data.answer) {
            bubble.innerHTML = formatMarkdown(data.answer);
            if (data.thread_id) {
              threadId = data.thread_id;
              localStorage.setItem("customer_agent_demo_thread_id", threadId);
            }
            if (data.perception || data.retrieved_docs) setState(data);
          }
        } catch {}
        return;
      }

      let fullAnswer = "";
      let finalState = null;
      let stateHandled = false;

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
            // Only update streaming bubble if we haven't received the final state yet.
            // After state is processed, subsequent tokens (hallucination check, etc.)
            // must NOT write to the now-detached streaming bubble or overwrite the final message.
            if (!stateHandled) {
              bubble.innerHTML = formatMarkdown(fullAnswer);
              messages.scrollTop = messages.scrollHeight;
            }
          } else if (event === "state") {
            finalState = parsed;
            
            // ── Immediately replace streaming bubble with final answer ──
            // Build meta tags
            const meta = [];
            if (parsed.perception?.intent) meta.push(parsed.perception.intent);
            if (parsed.perception?.emotion) meta.push(parsed.perception.emotion);
            if (parsed.active_agent) meta.push(parsed.active_agent);
            if (parsed.dialogue_status === "awaiting_clarification") meta.push("待澄清");
            
            // Replace streaming message with final (has references)
            streamMsg.remove();
            addMessage("agent", parsed.answer ?? fullAnswer, meta, {
              suggestions: parsed.clarification?.options || [],
            });
            
            setState(parsed);
            
            threadId = parsed.thread_id || threadId;
            localStorage.setItem("customer_agent_demo_thread_id", threadId);
            
            stateHandled = true;
          } else if (event === "error") {
            throw new Error(parsed.error || "流式响应错误");
          }
        }
      }
      
      if (stateHandled) {
        const freshSessions = await fetchSessions();
        if (freshSessions.length > 0 || !sessions.length) { sessions = freshSessions; }
        renderConversations();
      } else if (finalState) {
        bubble.innerHTML = formatMarkdown(finalState.answer ?? fullAnswer);
        const meta = [];
        if (finalState.perception?.intent) meta.push(finalState.perception.intent);
        if (finalState.perception?.emotion) meta.push(finalState.perception.emotion);
        if (finalState.active_agent) meta.push(finalState.active_agent);
        if (finalState.dialogue_status === "awaiting_clarification") meta.push("待澄清");
        streamMsg.remove();
        addMessage("agent", finalState.answer ?? fullAnswer, meta, {
          suggestions: finalState.clarification?.options || [],
        });
        setState(finalState);
        threadId = finalState.thread_id || threadId;
        localStorage.setItem("customer_agent_demo_thread_id", threadId);
        const freshSessions = await fetchSessions();
        if (freshSessions.length > 0 || !sessions.length) { sessions = freshSessions; }
        renderConversations();
      }
      // No else-if fallback needed: the pre-created streamMsg already exists with
      // whatever tokens were accumulated (fullAnswer). The typing dots were replaced
      // by answer_token events above.
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
    

