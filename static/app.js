    let sessions = [];
    let threadId = localStorage.getItem("customer_agent_demo_thread_id") || "";
    let userInitiatedNewThread = false;

    // Transitional: load legacy localStorage sessions if backend has none
    const LEGACY_SESSIONS_KEY = "customer_agent_demo_sessions";

    function loadLegacySessions() {
      try {
        const parsed = JSON.parse(localStorage.getItem(LEGACY_SESSIONS_KEY) || "[]");
        if (Array.isArray(parsed) && parsed.length) return parsed;
      } catch { console.warn("loadLegacySessions 解析失败"); }
      return [];
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

    if (messages && form && input && send && refs && defense && thread && newThread && conversations && searchThreads && toggleSidebar && toggleInspector && closeInspector && appLayout && toggleTheme && themeIcon) {
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

    newThread.addEventListener("click", async () => {
      userInitiatedNewThread = true;
      threadId = createThreadId();
      localStorage.setItem("customer_agent_demo_thread_id", threadId);
      // Immediately show the new conversation in the sidebar
      sessions.unshift({
        thread_id: threadId,
        title: "新对话",
        message_count: 0,
        created_at: new Date().toISOString(),
        last_updated: new Date().toISOString(),
      });
      renderConversations();
      await renderSession();
      input.focus();
    });

    async function initApp() {
      sessions = await fetchSessions();
      if (!sessions.length) {
        const legacy = loadLegacySessions();
        if (legacy.length) {
          sessions = legacy.map(s => ({
            thread_id: s.threadId,
            title: s.title,
            message_count: (s.messages || []).length,
            created_at: new Date(s.createdAt).toISOString(),
            last_updated: new Date(s.createdAt).toISOString(),
            _legacy: true,
            _messages: s.messages || [],
            _state: s.state || null,
          }));
        }
      }
      // Don't overwrite threadId if user just clicked "new thread" before initApp finished
      if (!userInitiatedNewThread) {
        if (!threadId && sessions.length > 0) {
          threadId = sessions[0].thread_id;
        }
        if (threadId && !sessions.some(s => s.thread_id === threadId)) {
          threadId = sessions.length > 0 ? sessions[0].thread_id : "";
        }
        if (threadId) {
          localStorage.setItem("customer_agent_demo_thread_id", threadId);
        }
      }
      await renderSession();
    }

    initApp();

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
    } // end of if (messages && ...) guard
