    function createThreadId() {
      const random = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2);
      return `web-${random}`;
    }

    async function fetchSessions() {
      try {
        const resp = await fetch("/api/conversations");
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.sessions || [];
      } catch { console.warn("fetchSessions 请求失败"); return []; }
    }

    async function fetchSessionMessages(threadId) {
      try {
        const resp = await fetch(`/api/conversations/${encodeURIComponent(threadId)}`);
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.messages || [];
      } catch { console.warn("fetchSessionMessages 请求失败"); return []; }
    }

