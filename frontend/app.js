(() => {
  const SESSION_KEY = "humphrey.sessionId";
  // The context bar measures against the window the server compacts the
  // chosen model to: that model's own from /api/models, or, for a model with
  // none, /healthz's fallback.
  let fallbackBudget = 128_000;
  let contextBudget = fallbackBudget;

  const stage = document.getElementById("stage");
  const thread = document.getElementById("thread");
  const transcript = document.getElementById("transcript");
  const composer = document.getElementById("composer");
  const question = document.getElementById("question");
  const primaryBtn = document.getElementById("primary-action");
  const primaryLabel = document.getElementById("primary-label");
  const stopBtn = document.getElementById("stop-btn");
  const followUpBtn = document.getElementById("follow-up-btn");
  const composerHint = document.getElementById("composer-hint");
  const newBtn = document.getElementById("new-chat");
  const newBtnCollapsed = document.getElementById("new-chat-collapsed");
  const statusLabel = document.getElementById("status-label");
  const modelSelect = /** @type {HTMLSelectElement | null} */ (document.getElementById("model-select"));
  const modelHint = document.getElementById("model-hint");
  const themeToggle = document.getElementById("theme-toggle");
  const detailsBtn = document.getElementById("details-btn");
  const sessionList = document.getElementById("session-list");
  const sessionsCount = document.getElementById("sessions-count");
  const queueEl = document.getElementById("queue");
  const toastEl = document.getElementById("toast");
  const srAnnounce = document.getElementById("sr-announce");
  const latestBtn = document.getElementById("latest-btn");
  const latestBtnDot = document.getElementById("latest-btn-dot");
  const scrim = document.getElementById("scrim");
  const statProvider = document.getElementById("stat-provider");
  const statStatus = document.getElementById("stat-status");
  const statTool = document.getElementById("stat-tool");
  const statTurns = document.getElementById("stat-turns");
  const statTools = document.getElementById("stat-tools");
  const statIo = document.getElementById("stat-io");
  const statContext = document.getElementById("stat-context");
  const contextBarFill = document.getElementById("context-bar-fill");
  const statsRingFill = document.getElementById("stats-ring-fill");
  const statsRingPct = document.getElementById("stats-ring-pct");
  const statsToggle = document.getElementById("stats-toggle");
  const attachBtn = document.getElementById("attach-btn");
  const attachInput = document.getElementById("attach-input");
  const composerAttachments = document.getElementById("composer-attachments");

  const MAX_ATTACHMENTS = 4;
  // Generous client-side pre-check; app/image_processing.py's 50MB source
  // ceiling is the real limit - this just avoids reading a huge file into
  // memory as a data URL for something the server would reject anyway.
  const MAX_ATTACHMENT_SOURCE_BYTES = 50 * 1024 * 1024;

  /** @type {{ role: "user" | "assistant"; content: string; images?: { data: string, mimeType: string }[] }[]} */
  let messages = [];
  /** @type {{ data: string, mimeType: string, previewUrl: string }[]} */
  let pendingAttachments = [];
  /** @type {string | null} */
  let sessionId = localStorage.getItem(SESSION_KEY);
  const PROJECT_KEY = "humphrey.project";
  /** The project in view: where new chats work, and whose chats and files
   * the rail lists. Null is the default project, the workspace itself.
   * @type {string | null} */
  let currentProject = localStorage.getItem(PROJECT_KEY);
  /** @type {{ root: string, projects: { name: string, folder: string, hidden: boolean }[], missing: string[] }} */
  let projectsInfo = { root: "", projects: [], missing: [] };
  /** @type {{ kind: "steer" | "follow-up"; text: string }[]} */
  let queue = [];
  let pinnedToBottom = true;
  let hasNewBelow = false;
  let pinFrame = 0;
  /** @type {AbortController | null} */
  let activeRun = null;
  let toolSeq = 0;
  let toolsUsed = 0;
  let charsIn = 0;
  let charsOut = 0;
  // This chat's usage, every model call added up (a turn that uses tools is
  // several calls). cachedIn is part of tokensIn, reasoningOut of tokensOut.
  let tokensIn = 0;
  let tokensOut = 0;
  let cachedIn = 0;
  let reasoningOut = 0;
  // The latest call alone: its input already holds the whole conversation,
  // so input + output is how full the context is now.
  let contextTokens = 0;
  /** @type {Map<string, any>} */
  let liveTools = new Map();
  /** @type {number | null} */
  let thoughtTimer = null;
  let toastTimer = 0;
  /** @type {string} */
  let lastToolSummary = "—";

  const PIN_THRESHOLD = 64;
  const RING_CIRC = 2 * Math.PI * 14;
  const THOUGHT_WORDS = ["Thinking", "Pondering", "Considering", "Reflecting"];
  const THOUGHT_WORD_MS = 3200;
  const md = window.HumphreyMarkdown;
  /** @type {Promise<any> | null} */
  let mermaidLoader = null;
  let mermaidSeq = 0;
  const ASSET_QUERY = (() => {
    try {
      const src = document.currentScript && document.currentScript.src;
      if (!src) {
        return "";
      }
      const version = new URL(src, window.location.href).searchParams.get("v");
      return version ? `?v=${encodeURIComponent(version)}` : "";
    } catch {
      return "";
    }
  })();

  function setSessionsOpen(open) {
    stage.dataset.sessionsOpen = open ? "1" : "0";
    if (open) {
      void loadFiles();
    }
    const toggle = document.getElementById("sessions-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }
    syncScrim();
  }

  function setStatsOpen(open) {
    stage.dataset.statsOpen = open ? "1" : "0";
    for (const toggle of [document.getElementById("stats-toggle"), detailsBtn]) {
      toggle?.setAttribute("aria-expanded", open ? "true" : "false");
    }
    syncScrim();
  }

  function syncScrim() {
    if (!scrim) {
      return;
    }
    const open = stage.dataset.sessionsOpen === "1" || stage.dataset.statsOpen === "1";
    scrim.hidden = !open;
  }

  /** @param {"connecting" | "ready" | "working"} status */
  function setStatus(status) {
    stage.dataset.status = status;
    const labels = { connecting: "Connecting", ready: "Ready", working: "Working", waiting: "Needs you" };
    if (statusLabel) {
      statusLabel.textContent = labels[status];
    }
    if (statStatus) {
      statStatus.textContent = labels[status];
    }
  }

  function setActive(active) {
    stage.dataset.state = active ? "active" : "empty";
    updateLatestButton();
  }

  function setRunning(running) {
    stage.dataset.running = running ? "1" : "0";
    primaryLabel.textContent = running ? "Steer" : "Send";
    primaryBtn.setAttribute("aria-label", running ? "Steer the current turn" : "Send message");
    stopBtn.hidden = !running;
    followUpBtn.hidden = !running;
    // Steer/follow-up (mid-run) are brief text nudges, not attachment points.
    if (attachBtn) {
      attachBtn.disabled = running;
    }
    question.placeholder = running ? "Steer the current turn…" : "Message Humphrey…";
    if (composerHint) {
      composerHint.innerHTML = running
        ? "<kbd>Enter</kbd> steer · <kbd>Alt</kbd>+<kbd>Enter</kbd> follow-up · <kbd>Esc</kbd> stop"
        : "<kbd>Enter</kbd> send · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline";
    }
    syncSendEnabled();
    setStatus(running ? "working" : "ready");
  }

  function syncSendEnabled() {
    const hasText = question.value.trim().length > 0;
    primaryBtn.disabled = !hasText && pendingAttachments.length === 0;
  }

  /**
   * Read a File as a data URL, split it into raw base64 + mime type. The
   * server never trusts this mime type for decoding (app/image_processing.py
   * sniffs magic bytes instead) - it's only used here for the preview <img>
   * and an early "is this even an image" filter.
   * @param {File} file
   * @returns {Promise<{ data: string, mimeType: string, previewUrl: string } | null>}
   */
  function readFileAsAttachment(file) {
    return new Promise((resolve) => {
      if (!file.type.startsWith("image/")) {
        resolve(null);
        return;
      }
      if (file.size > MAX_ATTACHMENT_SOURCE_BYTES) {
        showToast(`${file.name || "Image"} is too large (max 50MB)`);
        resolve(null);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const result = typeof reader.result === "string" ? reader.result : "";
        const match = /^data:([^;]+);base64,(.*)$/s.exec(result);
        if (!match) {
          resolve(null);
          return;
        }
        resolve({ data: match[2], mimeType: match[1], previewUrl: result });
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }

  /** @param {FileList | File[]} files */
  async function addAttachments(files) {
    for (const file of Array.from(files)) {
      if (pendingAttachments.length >= MAX_ATTACHMENTS) {
        showToast(`Up to ${MAX_ATTACHMENTS} images per message`);
        break;
      }
      const attachment = await readFileAsAttachment(file);
      if (attachment) {
        pendingAttachments.push(attachment);
      }
    }
    renderAttachmentPreviews();
    syncSendEnabled();
  }

  function removeAttachment(index) {
    pendingAttachments.splice(index, 1);
    renderAttachmentPreviews();
    syncSendEnabled();
  }

  function renderAttachmentPreviews() {
    if (!composerAttachments) {
      return;
    }
    composerAttachments.replaceChildren();
    composerAttachments.hidden = pendingAttachments.length === 0;
    pendingAttachments.forEach((attachment, index) => {
      const chip = document.createElement("div");
      chip.className = "attachment-chip";
      const img = document.createElement("img");
      img.src = attachment.previewUrl;
      img.alt = "Attached image";
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "attachment-chip-remove";
      remove.setAttribute("aria-label", "Remove image");
      remove.textContent = "×";
      remove.addEventListener("click", () => removeAttachment(index));
      chip.append(img, remove);
      composerAttachments.append(chip);
    });
  }

  /** @param {string} text */
  function announce(text) {
    if (!srAnnounce) {
      return;
    }
    srAnnounce.textContent = "";
    requestAnimationFrame(() => {
      srAnnounce.textContent = text;
    });
  }

  /** @param {string} text */
  function showToast(text) {
    if (!toastEl) {
      return;
    }
    toastEl.hidden = false;
    toastEl.textContent = text;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toastEl.hidden = true;
    }, 3200);
  }

  // ── Model picker ─────────────────────────────────────────
  // The chosen model is a standing preference (localStorage), sent with
  // every turn, so switching mid-conversation applies from the next message.
  const MODEL_KEY = "humphrey.model";
  /** @type {string | null} null = the provider decides (mock, Azure) */
  let selectedModel = null;

  // Reasoning effort: a standing preference like the model, but checked
  // against what the chosen model offers each time it changes.
  const EFFORT_KEY = "humphrey.effort";
  const EFFORT_LABELS = { low: "Low", medium: "Medium", high: "High", xhigh: "Extra high", max: "Max" };
  const effortRow = document.getElementById("effort-row");
  const effortSelect = /** @type {HTMLSelectElement | null} */ (document.getElementById("effort-select"));
  /** @type {string | null} null = the server's default (REASONING_EFFORT) */
  let selectedEffort = null;
  /** @type {{ efforts?: Record<string, string[]>, default_effort?: string | null, default: string,
   *   context_windows?: Record<string, number> } | null} */
  let modelsInfo = null;

  /** Point the context bar at the chosen model's window. */
  function applyContextBudget() {
    const model = selectedModel || (modelsInfo && modelsInfo.default);
    const own = modelsInfo && modelsInfo.context_windows && model ? modelsInfo.context_windows[model] : 0;
    contextBudget = Number(own) > 0 ? Number(own) : fallbackBudget;
    updateStats();
  }

  function renderEffortPicker() {
    if (!effortRow || !effortSelect || !modelsInfo) {
      return;
    }
    const model = selectedModel || modelsInfo.default;
    const levels = (modelsInfo.efforts && modelsInfo.efforts[model]) || [];
    effortRow.hidden = levels.length === 0;
    if (levels.length === 0) {
      selectedEffort = null;
      return;
    }
    const fallback = modelsInfo.default_effort && levels.includes(modelsInfo.default_effort)
      ? modelsInfo.default_effort
      : levels[0];
    const stored = readStored(EFFORT_KEY);
    selectedEffort = stored && levels.includes(stored) ? stored : fallback;
    effortSelect.replaceChildren(
      ...levels.map((level) => {
        const option = document.createElement("option");
        option.value = level;
        const name = EFFORT_LABELS[level] || level;
        option.textContent = level === modelsInfo.default_effort ? `${name} · default` : name;
        return option;
      }),
    );
    effortSelect.value = selectedEffort;
  }

  effortSelect?.addEventListener("change", () => {
    selectedEffort = effortSelect.value;
    writeStored(EFFORT_KEY, selectedEffort);
    const name = EFFORT_LABELS[selectedEffort] || selectedEffort;
    showToast(
      stage.dataset.running === "1"
        ? `Reasoning set to ${name} from your next message`
        : `Reasoning set to ${name}`,
    );
  });

  function readStored(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function writeStored(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* private mode: the choice just won't persist */
    }
  }

  /** @param {{ provider: string, selectable: boolean, default: string, models: string[],
   *   labels?: Record<string, string> }} info */
  function renderModelPicker(info) {
    if (!modelSelect) {
      return;
    }
    const models = info.models.length ? info.models : [info.default];
    const stored = readStored(MODEL_KEY);
    selectedModel = info.selectable ? (stored && models.includes(stored) ? stored : info.default) : null;
    modelSelect.replaceChildren(
      ...models.map((m) => {
        const option = document.createElement("option");
        option.value = m;
        const name = (info.labels && info.labels[m]) || m;
        option.textContent =
          m === info.default && info.selectable && models.length > 1 ? `${name} · default` : name;
        return option;
      }),
    );
    modelSelect.value = selectedModel || info.default;
    modelSelect.disabled = !info.selectable || models.length < 2;
    modelsInfo = info;
    renderEffortPicker();
    applyContextBudget();
    modelSelect.title = modelSelect.value;
    if (modelHint) {
      const hints = {
        mock: "Offline mock: no model is called.",
        azure_openai: "Set by the Azure deployment.",
        unconfigured: "Set OPENAI_API_KEY in .env.",
      };
      // The plan's list comes from the account, so OPENAI_MODELS doesn't apply.
      const hint = info.provider === "chatgpt"
        ? ""
        : info.selectable
        ? models.length < 2
          ? "Add more with OPENAI_MODELS in .env."
          : ""
        : hints[info.provider] || "";
      modelHint.textContent = hint;
      modelHint.hidden = !hint;
    }
  }

  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        renderModelPicker(await res.json());
      }
    } catch {
      /* leave the placeholder; the next probe retries */
    }
  }

  modelSelect?.addEventListener("change", () => {
    selectedModel = modelSelect.value;
    modelSelect.title = selectedModel;
    writeStored(MODEL_KEY, selectedModel);
    renderEffortPicker(); // the new model may offer different levels
    applyContextBudget(); // and have a different context window
    showToast(
      stage.dataset.running === "1"
        ? `Switched to ${selectedModel} from your next message`
        : `Switched to ${selectedModel}`,
    );
  });

  // ── Model access: API key or ChatGPT plan ────────────────
  // Signing in happens on OpenAI's site in another tab; this tab learns the
  // result from /api/auth/status. Tokens stay on the server: nothing here
  // ever sees one.
  const MANAGE_USAGE_URL = "https://chatgpt.com/settings/usage";
  const accessRow = document.getElementById("access-row");
  const accessChoice = document.getElementById("access-choice");
  const accessHint = document.getElementById("access-hint");
  const accessWarning = document.getElementById("access-warning");
  const signInBtn = document.getElementById("chatgpt-sign-in");
  const signOutBtn = document.getElementById("chatgpt-sign-out");
  const planIndicator = document.getElementById("plan-indicator");
  const planNotice = /** @type {HTMLDialogElement | null} */ (document.getElementById("plan-notice"));
  /** @type {number | null} */
  let signInPoll = null;

  /** POST JSON: the server refuses anything else, so no other site can trigger these. */
  async function postJson(path, body = {}) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
    }
    return data;
  }

  /**
   * @param {{ signed_in: boolean, email: string | null, active_source: string,
   *   using_plan: boolean, plan_notice_seen: boolean, credentials_private: boolean }} status
   * @param {string} provider the current source from /healthz
   */
  function renderAccess(status, provider) {
    if (!accessRow) {
      return;
    }
    // The offline mock answers everything, so a sign-in would do nothing.
    accessRow.hidden = provider === "mock";
    if (signInBtn) {
      signInBtn.hidden = status.signed_in;
    }
    if (accessChoice) {
      accessChoice.hidden = !status.signed_in;
      for (const button of accessChoice.querySelectorAll("button")) {
        button.setAttribute("aria-checked", String(button.dataset.source === status.active_source));
      }
    }
    if (signOutBtn) {
      signOutBtn.hidden = !status.signed_in;
    }
    if (accessHint) {
      accessHint.textContent = status.signed_in
        ? `Signed in as ${status.email || "your ChatGPT account"}`
        : "Use your ChatGPT plan instead of an API key.";
    }
    if (accessWarning) {
      accessWarning.hidden = !(status.signed_in && status.credentials_private === false);
    }
    if (planIndicator) {
      planIndicator.hidden = !status.using_plan;
    }
    if (status.using_plan && !status.plan_notice_seen && planNotice && !planNotice.open) {
      planNotice.showModal();
    }
  }

  async function loadAuth(provider) {
    try {
      const res = await fetch("/api/auth/status");
      if (res.ok) {
        renderAccess(await res.json(), provider);
      }
    } catch {
      /* the next probe retries */
    }
  }

  /** Re-read everything the source affects: access, models, provider label. */
  async function refreshAccess() {
    await probeHealth();
  }

  function stopSignInPoll() {
    if (signInPoll !== null) {
      clearInterval(signInPoll);
      signInPoll = null;
    }
    if (signInBtn) {
      signInBtn.disabled = false;
      signInBtn.textContent = "Continue with ChatGPT";
    }
  }

  signInBtn?.addEventListener("click", async () => {
    stopSignInPoll(); // a second click replaces any sign-in still waiting
    // Open the tab now, inside the click, or a popup blocker stops it; it's
    // pointed at OpenAI once the server has made the sign-in URL. Cutting
    // opener keeps OpenAI's pages from reaching back into this window.
    const tab = window.open("about:blank", "_blank");
    if (tab) {
      tab.opener = null;
    }
    signInBtn.disabled = true;
    signInBtn.textContent = "Waiting for ChatGPT…";
    let url;
    try {
      ({ url } = await postJson("/api/auth/chatgpt/start"));
    } catch (err) {
      tab?.close();
      stopSignInPoll();
      showToast(`Couldn't start sign-in: ${err instanceof Error ? err.message : err}`);
      return;
    }
    if (tab) {
      tab.location.href = url;
    } else {
      window.location.href = url; // no new tab allowed: sign in here, then come back
      return;
    }
    const startedAt = Date.now();
    signInPoll = window.setInterval(async () => {
      if (Date.now() - startedAt > 10 * 60 * 1000) {
        stopSignInPoll();
        return;
      }
      try {
        const status = await (await fetch("/api/auth/status")).json();
        if (status.signed_in) {
          stopSignInPoll();
          showToast(`Signed in as ${status.email || "your ChatGPT account"}`);
          await refreshAccess();
        }
      } catch {
        /* keep waiting */
      }
    }, 2000);
  });

  accessChoice?.addEventListener("click", async (event) => {
    const button = event.target instanceof Element ? event.target.closest("button[data-source]") : null;
    if (!(button instanceof HTMLButtonElement) || button.getAttribute("aria-checked") === "true") {
      return;
    }
    try {
      await postJson("/api/auth/source", { source: button.dataset.source });
      showToast(button.dataset.source === "chatgpt" ? "Using your ChatGPT plan" : "Using your API key");
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err));
    }
    await refreshAccess();
  });

  signOutBtn?.addEventListener("click", async () => {
    try {
      await postJson("/api/auth/chatgpt/sign-out");
      showToast("Signed out of ChatGPT");
    } catch (err) {
      showToast(err instanceof Error ? err.message : String(err));
    }
    await refreshAccess();
  });

  document.getElementById("plan-notice-ok")?.addEventListener("click", async () => {
    planNotice?.close();
    try {
      await postJson("/api/auth/chatgpt/plan-notice-seen");
    } catch {
      /* it shows again next time; harmless */
    }
  });

  // ── Theme ────────────────────────────────────────────────
  const THEME_KEY = "humphrey.theme";
  const THEME_COLOR = { dark: "#161826", light: "#f6f4f0" };

  function currentTheme() {
    return document.documentElement.dataset.theme === "light" ? "light" : "dark";
  }

  /** @param {"light" | "dark"} theme */
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", THEME_COLOR[theme]);
    if (themeToggle) {
      const next = theme === "light" ? "dark" : "light";
      const label = `Switch to ${next} theme`;
      themeToggle.setAttribute("aria-label", label);
      themeToggle.title = label;
      themeToggle.innerHTML = `<i class="ph ph-${next === "light" ? "sun" : "moon"}" aria-hidden="true"></i>`;
    }
    void rethemeMermaid();
    syncArtifactTheme();
  }

  themeToggle?.addEventListener("click", () => {
    const next = currentTheme() === "light" ? "dark" : "light";
    writeStored(THEME_KEY, next);
    applyTheme(next);
  });

  // Follow the OS until the user has made an explicit choice.
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (event) => {
    const stored = readStored(THEME_KEY);
    if (stored !== "light" && stored !== "dark") {
      applyTheme(event.matches ? "light" : "dark");
    }
  });

  /** @param {string} summary */
  function setToolStat(summary) {
    lastToolSummary = summary || "—";
    if (statTool) {
      statTool.textContent = lastToolSummary;
      statTool.title = lastToolSummary;
    }
  }

  function distanceFromBottom() {
    return thread.scrollHeight - thread.scrollTop - thread.clientHeight;
  }

  function updateLatestButton() {
    if (!latestBtn) {
      return;
    }
    const empty = stage.dataset.state !== "active";
    stage.dataset.pinned = pinnedToBottom ? "1" : "0";
    const show = !empty && !pinnedToBottom;
    const wasHidden = latestBtn.hidden;
    latestBtn.hidden = !show;
    if (!show) {
      hasNewBelow = false;
      latestBtn.classList.remove("is-new");
      if (latestBtnDot) {
        latestBtnDot.hidden = true;
      }
      return;
    }
    if (wasHidden) {
      announce("Latest messages available");
    }
    latestBtn.classList.toggle("is-new", hasNewBelow);
    if (latestBtnDot) {
      latestBtnDot.hidden = !hasNewBelow;
    }
  }

  function syncPinFromScroll() {
    pinnedToBottom = distanceFromBottom() <= PIN_THRESHOLD;
    if (pinnedToBottom) {
      hasNewBelow = false;
    }
    updateLatestButton();
  }

  /** Signal growth below the viewport when the user has scrolled up. */
  function noteContentGrowth() {
    if (!pinnedToBottom) {
      hasNewBelow = true;
      updateLatestButton();
    }
  }

  function pinToBottom() {
    if (!pinnedToBottom) {
      noteContentGrowth();
      return;
    }
    cancelAnimationFrame(pinFrame);
    pinFrame = requestAnimationFrame(() => {
      thread.scrollTop = thread.scrollHeight;
      hasNewBelow = false;
      updateLatestButton();
    });
  }

  function jumpToLatest() {
    pinnedToBottom = true;
    hasNewBelow = false;
    thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
    updateLatestButton();
  }

  function autoSize() {
    question.style.height = "auto";
    const next = Math.min(Math.max(question.scrollHeight, 56), 220);
    question.style.height = `${next}px`;
    syncSendEnabled();
    pinToBottom();
  }

  function clearTranscript() {
    transcript.replaceChildren();
    artifactCards.clear();
  }

  /**
   * @param {HTMLElement} el
   * @param {string} text
   * @param {{ caret?: boolean }} [opts]
   */
  function setMarkdown(el, text, opts = {}) {
    if (md && typeof md.render === "function") {
      el.innerHTML = md.render(text);
      bindCodeCopy(el);
    } else {
      el.textContent = text;
    }
    if (opts.caret) {
      attachStreamCaret(el);
    } else {
      void hydrateMermaid(el);
    }
  }

  // ── Artefacts ────────────────────────────────────────────
  // Files the agent creates or changes in its workspace, rendered inline in
  // the transcript. The server sends an "artifact" event per change and
  // serves the file from /api/files (sandboxed); see app/artifacts.py.

  const ARTIFACT_ICONS = {
    html: "ph-browser",
    markdown: "ph-file-text",
    image: "ph-image",
    table: "ph-table",
    diagram: "ph-graph",
    code: "ph-file-code",
    pdf: "ph-file-pdf",
    file: "ph-file",
  };
  const ARTIFACT_LABELS = {
    html: "HTML",
    markdown: "Markdown",
    image: "Image",
    table: "Table",
    diagram: "Diagram",
    code: "Code",
    pdf: "PDF",
    file: "File",
  };
  const SOURCE_LANG = { html: "html", markdown: "markdown", table: "csv", diagram: "mermaid" };
  const MAX_TEXT_PREVIEW_BYTES = 1_000_000;
  const MAX_TABLE_ROWS = 200;
  /** Latest card per path in the visible transcript, so a new version can retire the old one. */
  const artifactCards = new Map();
  const artifactViewer = /** @type {HTMLDialogElement | null} */ (document.getElementById("artifact-viewer"));

  /** @param {string} path */
  function fileUrl(path, params = {}) {
    const encoded = path.split("/").map(encodeURIComponent).join("/");
    const query = new URLSearchParams(params).toString();
    return `/api/files/${encoded}${query ? `?${query}` : ""}`;
  }

  /** @param {number} bytes */
  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  /** @param {string} path */
  function extensionOf(path) {
    const match = /\.([a-z0-9]+)$/i.exec(path);
    return match ? match[1].toLowerCase() : "";
  }

  /** Minimal RFC 4180 parser: quotes, escaped quotes, newlines in fields. */
  function parseDelimited(text, delimiter) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else if (ch === '"') {
          quoted = false;
        } else {
          field += ch;
        }
      } else if (ch === '"') {
        quoted = true;
      } else if (ch === delimiter) {
        row.push(field);
        field = "";
      } else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i += 1;
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else {
        field += ch;
      }
    }
    if (field || row.length) {
      row.push(field);
      rows.push(row);
    }
    return rows;
  }

  /** @param {string} text @param {string} lang */
  function codeBlock(text, lang) {
    const host = document.createElement("div");
    host.className = "msg-body artifact-code";
    const fence = text.includes("```") ? "````" : "```";
    setMarkdown(host, `${fence}${lang}\n${text.replace(/\n$/, "")}\n${fence}`);
    return host;
  }

  /** @param {{ path: string, size?: number }} art */
  async function fetchArtifactText(art) {
    if ((art.size ?? 0) > MAX_TEXT_PREVIEW_BYTES) {
      throw new Error("too-large");
    }
    const res = await fetch(fileUrl(art.path, { v: String(art.mtime ?? "") }));
    if (!res.ok) throw new Error(res.status === 404 ? "missing" : "failed");
    return res.text();
  }

  /** @param {string} message */
  function previewNote(message) {
    const note = document.createElement("p");
    note.className = "artifact-note";
    note.textContent = message;
    return note;
  }

  /**
   * Fill `body` with the artefact's preview.
   * @param {HTMLElement} body
   * @param {any} art
   * @param {{ full?: boolean, source?: boolean }} opts
   */
  async function renderArtifactPreview(body, art, opts = {}) {
    body.replaceChildren();
    body.dataset.kind = art.kind;
    try {
      if (opts.source) {
        const lang = SOURCE_LANG[art.kind] || extensionOf(art.path) || "text";
        body.append(codeBlock(await fetchArtifactText(art), lang));
        return;
      }
      if (art.kind === "html") {
        const frame = document.createElement("iframe");
        frame.className = "artifact-frame";
        frame.title = shownPath(art);
        frame.setAttribute("sandbox", "allow-scripts allow-popups");
        frame.setAttribute("loading", "lazy");
        frame.src = fileUrl(art.path, { embed: "true", theme: currentTheme(), v: String(art.mtime ?? "") });
        if (opts.full) frame.dataset.full = "1";
        body.append(frame);
        return;
      }
      if (art.kind === "image") {
        const img = document.createElement("img");
        img.className = "artifact-image";
        img.alt = shownPath(art);
        img.loading = "lazy";
        img.src = fileUrl(art.path, { v: String(art.mtime ?? "") });
        if (!opts.full) img.addEventListener("click", () => openArtifactViewer(art));
        body.append(img);
        return;
      }
      if (art.kind === "pdf") {
        const frame = document.createElement("iframe");
        frame.className = "artifact-frame artifact-pdf";
        frame.title = shownPath(art);
        frame.src = fileUrl(art.path, { v: String(art.mtime ?? "") });
        body.append(frame);
        return;
      }
      if (art.kind === "file") {
        body.append(previewNote("No preview for this file type. Download it to open."));
        return;
      }
      const text = await fetchArtifactText(art);
      if (art.kind === "markdown") {
        const doc = document.createElement("div");
        doc.className = "msg-body artifact-doc";
        setMarkdown(doc, text);
        body.append(doc);
      } else if (art.kind === "diagram") {
        const doc = document.createElement("div");
        doc.className = "msg-body artifact-doc";
        setMarkdown(doc, `\`\`\`mermaid\n${text}\n\`\`\``);
        body.append(doc);
      } else if (art.kind === "table") {
        body.append(renderTable(text, extensionOf(art.path) === "tsv" ? "\t" : ","));
      } else {
        body.append(codeBlock(text, extensionOf(art.path) || "text"));
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : "";
      body.replaceChildren(
        previewNote(
          reason === "too-large"
            ? "Too large to preview. Download it to open."
            : reason === "missing"
              ? "This file no longer exists in the workspace."
              : "Could not load a preview.",
        ),
      );
    }
    if (!opts.full) fitCollapsible(body);
  }

  /** @param {string} text @param {string} delimiter */
  function renderTable(text, delimiter) {
    const rows = parseDelimited(text, delimiter).filter((r) => r.some((c) => c.trim()));
    const wrap = document.createElement("div");
    wrap.className = "msg-body artifact-table";
    if (!rows.length) {
      wrap.append(previewNote("Empty table."));
      return wrap;
    }
    const numeric = /^[-+]?[£$€]?\d[\d,]*(\.\d+)?[%kmb]?$/i;
    const body = rows.slice(1, MAX_TABLE_ROWS + 1);
    // A column is numeric when most of its filled cells are: those align
    // right (header included) in tabular tan mono.
    const numericCols = rows[0].map((_, col) => {
      const filled = body.map((r) => (r[col] ?? "").trim()).filter(Boolean);
      return filled.length > 0 && filled.filter((v) => numeric.test(v)).length / filled.length >= 0.8;
    });
    const table = document.createElement("table");
    const thead = table.createTHead().insertRow();
    rows[0].forEach((cell, col) => {
      const th = document.createElement("th");
      th.textContent = cell;
      if (numericCols[col]) th.className = "is-num";
      thead.append(th);
    });
    const tbody = table.createTBody();
    for (const r of body) {
      const tr = tbody.insertRow();
      r.forEach((cell, col) => {
        const td = tr.insertCell();
        td.textContent = cell;
        if (numericCols[col]) td.className = "is-num";
      });
    }
    wrap.append(table);
    if (rows.length - 1 > MAX_TABLE_ROWS) {
      wrap.append(previewNote(`Showing ${MAX_TABLE_ROWS} of ${rows.length - 1} rows.`));
    }
    return wrap;
  }

  /** Long previews start clipped with a fade and a "Show all" control. */
  function fitCollapsible(body) {
    requestAnimationFrame(() => {
      if (body.scrollHeight <= body.clientHeight + 8 || body.dataset.kind === "html") return;
      body.classList.add("is-clipped");
      const more = document.createElement("button");
      more.type = "button";
      more.className = "artifact-more";
      more.textContent = "Show all";
      more.addEventListener("click", () => {
        body.classList.remove("is-clipped");
        more.remove();
      });
      body.after(more);
    });
  }

  /** @param {string} icon @param {string} label */
  function artifactAction(icon, label) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "header-icon-btn artifact-action";
    btn.setAttribute("aria-label", label);
    btn.title = label;
    btn.innerHTML = `<i class="ph ${icon}" aria-hidden="true"></i>`;
    return btn;
  }

  /**
   * Header shared by inline cards and the fullscreen viewer.
   * @param {any} art
   * @param {{ onSource?: (on: boolean) => void, onExpand?: () => void, onClose?: () => void }} handlers
   */
  function artifactHeader(art, handlers) {
    const head = document.createElement("header");
    head.className = "artifact-head";

    const icon = document.createElement("span");
    icon.className = "artifact-icon";
    icon.innerHTML = `<i class="ph ${ARTIFACT_ICONS[art.kind] || "ph-file"}" aria-hidden="true"></i>`;

    const title = document.createElement("div");
    title.className = "artifact-title";
    const name = document.createElement("span");
    name.className = "artifact-name";
    name.textContent = shownPath(art);
    const meta = document.createElement("span");
    meta.className = "artifact-meta";
    meta.textContent = [ARTIFACT_LABELS[art.kind] || "File", formatBytes(art.size)].filter(Boolean).join(" · ");
    title.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "artifact-actions";
    if (handlers.onSource && SOURCE_LANG[art.kind]) {
      const source = artifactAction("ph-code", "Show source");
      source.setAttribute("aria-pressed", "false");
      source.addEventListener("click", () => {
        const on = source.getAttribute("aria-pressed") !== "true";
        source.setAttribute("aria-pressed", on ? "true" : "false");
        source.setAttribute("aria-label", on ? "Show preview" : "Show source");
        source.title = on ? "Show preview" : "Show source";
        handlers.onSource?.(on);
      });
      actions.append(source);
    }
    const open = artifactAction("ph-arrow-square-out", "Open in new tab");
    open.addEventListener("click", () => window.open(fileUrl(art.path), "_blank", "noopener"));
    const download = artifactAction("ph-download-simple", "Download");
    download.addEventListener("click", () => {
      const a = document.createElement("a");
      a.href = fileUrl(art.path, { download: "true" });
      a.download = art.path.split("/").pop() || "download";
      a.click();
    });
    actions.append(open, download);
    if (handlers.onExpand) {
      const expand = artifactAction("ph-arrows-out", "Expand");
      expand.addEventListener("click", handlers.onExpand);
      actions.append(expand);
    }
    if (handlers.onClose) {
      const close = artifactAction("ph-x", "Close");
      close.addEventListener("click", handlers.onClose);
      actions.append(close);
    }

    head.append(icon, title);
    if ((art.version ?? 1) > 1) {
      const version = document.createElement("span");
      version.className = "artifact-version";
      version.textContent = `v${art.version}`;
      version.title = `Version ${art.version} of this file in this chat`;
      head.append(version);
    }
    head.append(actions);
    return head;
  }

  /**
   * An inline artefact card. Code files start collapsed; everything else
   * previews straight away.
   * @param {any} art
   * @param {{ animate?: boolean }} [opts]
   */
  function createArtifactCard(art, opts = {}) {
    const card = document.createElement("section");
    card.className = "artifact";
    if (opts.animate !== false) card.classList.add("msg-enter");
    card.dataset.kind = art.kind;
    card.setAttribute("aria-label", `${ARTIFACT_LABELS[art.kind] || "File"} artefact ${shownPath(art)}`);

    const body = document.createElement("div");
    body.className = "artifact-body";
    let source = false;
    const head = artifactHeader(art, {
      onSource: (on) => {
        source = on;
        card.querySelector(".artifact-more")?.remove();
        void renderArtifactPreview(body, art, { source });
      },
      onExpand: () => openArtifactViewer(art),
    });
    card.append(head, body);

    if (art.kind === "code") {
      card.dataset.collapsed = "1";
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "artifact-toggle";
      toggle.setAttribute("aria-expanded", "false");
      toggle.innerHTML = '<i class="ph ph-caret-down" aria-hidden="true"></i><span>Show code</span>';
      toggle.addEventListener("click", () => {
        const open = card.dataset.collapsed === "1";
        card.dataset.collapsed = open ? "0" : "1";
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        toggle.querySelector("span").textContent = open ? "Hide code" : "Show code";
        if (open && !body.childElementCount) void renderArtifactPreview(body, art);
      });
      head.querySelector(".artifact-title")?.after(toggle);
    } else {
      void renderArtifactPreview(body, art);
    }
    return card;
  }

  /** Mark an earlier card for the same file as superseded: its preview would show the new content. */
  function retireArtifactCard(path, version) {
    const previous = artifactCards.get(path);
    if (!previous) return;
    previous.classList.add("is-superseded");
    previous.querySelector(".artifact-body")?.replaceChildren();
    previous.querySelector(".artifact-more")?.remove();
    const meta = previous.querySelector(".artifact-meta");
    if (meta) meta.textContent = `Updated below to v${version}`;
  }

  /**
   * Place an artefact card: at the end of the live transcript, or before
   * `before` when rebuilding a reopened session.
   * @param {any} art
   * @param {{ animate?: boolean, before?: Element | null }} [opts]
   */
  function placeArtifact(art, opts = {}) {
    retireArtifactCard(art.path, art.version ?? 1);
    const card = createArtifactCard(art, opts);
    if (opts.before) {
      transcript.insertBefore(card, opts.before);
    } else {
      transcript.append(card);
    }
    artifactCards.set(art.path, card);
    return card;
  }

  /** @param {any} art */
  function openArtifactViewer(art) {
    if (!artifactViewer) return;
    const body = document.createElement("div");
    body.className = "artifact-body is-full";
    const close = () => artifactViewer.close();
    const head = artifactHeader(art, {
      onSource: (on) => void renderArtifactPreview(body, art, { full: true, source: on }),
      onClose: close,
    });
    artifactViewer.replaceChildren(head, body);
    void renderArtifactPreview(body, art, { full: true });
    artifactViewer.showModal();
  }

  artifactViewer?.addEventListener("click", (event) => {
    if (event.target === artifactViewer) artifactViewer.close(); // backdrop click
  });
  artifactViewer?.addEventListener("close", () => artifactViewer.replaceChildren());

  // HTML artefacts report their height from inside the sandbox.
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.humphrey !== "artifact-size" || typeof data.height !== "number") return;
    for (const frame of document.querySelectorAll("iframe.artifact-frame")) {
      if (frame.contentWindow === event.source && frame.dataset.full !== "1") {
        frame.style.height = `${Math.min(Math.max(data.height, 120), 880)}px`;
      }
    }
  });

  function syncArtifactTheme() {
    for (const frame of document.querySelectorAll("iframe.artifact-frame:not(.artifact-pdf)")) {
      frame.contentWindow?.postMessage({ humphrey: "theme", theme: currentTheme() }, "*");
    }
  }

  // ── Files list (sessions rail) ───────────────────────────
  const filesList = document.getElementById("files-list");

  async function loadFiles() {
    if (!filesList) return;
    const project = currentProject;
    try {
      const res = await fetch(`/api/files${projectQuery()}`);
      if (project !== currentProject) return; // a switch overtook this answer
      if (res.status === 404) {
        filesList.replaceChildren(missingFolderState());
        return;
      }
      if (!res.ok) return;
      const files = await res.json();
      if (project !== currentProject) return;
      filesList.replaceChildren();
      if (!files.length) {
        filesList.append(filesEmptyState());
        return;
      }
      for (const file of files.slice(0, 50)) {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "session-item file-item";
        btn.innerHTML = `<i class="ph ${ARTIFACT_ICONS[file.kind] || "ph-file"}" aria-hidden="true"></i>`;
        const name = document.createElement("span");
        name.className = "session-title";
        name.textContent = shownPath(file);
        btn.append(name);
        btn.title = `${shownPath(file)} · ${formatBytes(file.size)}`;
        btn.addEventListener("click", () => openArtifactViewer(file));
        li.append(btn);
        filesList.append(li);
      }
    } catch {
      /* the list just stays as it was */
    }
  }

  /** The project's folder has been renamed or removed on disk. */
  function missingFolderState() {
    const li = document.createElement("li");
    li.className = "files-empty";
    li.textContent = `Its folder, projects/${currentProject}/, isn't there any more. Its chats can still be read.`;
    return li;
  }

  /** No files yet: say where the person's own files go, ready to copy. */
  function filesEmptyState() {
    const li = document.createElement("li");
    li.className = "files-empty";
    const note = document.createElement("span");
    note.textContent = "No files yet. Put your own files here and Humphrey can use them:";
    li.append(note);
    const folder = projectFolder();
    if (folder) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "path-chip";
      chip.title = "Copy the folder's path";
      const text = document.createElement("span");
      text.textContent = folder;
      chip.append(text);
      chip.insertAdjacentHTML("beforeend", '<i class="ph ph-copy" aria-hidden="true"></i>');
      chip.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(folder);
          showToast("Copied the folder's path");
        } catch {
          window.getSelection()?.selectAllChildren(text);
        }
      });
      li.append(chip);
    }
    return li;
  }

  // Projects (app/projects.py): a folder in the workspace and the chats
  // that work in it. The switcher at the top of the rail picks the one in
  // view; opening a chat brings its project into view.
  const projectButton = document.getElementById("project-button");
  const projectMenu = document.getElementById("project-menu");
  const projectList = document.getElementById("project-list");
  const projectName = document.getElementById("project-name");
  const projectNewBtn = document.getElementById("project-new-btn");
  const projectRenameBtn = document.getElementById("project-rename-btn");
  const projectHideBtn = document.getElementById("project-hide-btn");
  const projectFormLabel = document.getElementById("project-form-label");
  const projectFormSubmit = document.getElementById("project-new-create");
  /** Whether the menu lists hidden projects, until it next opens. */
  let showHidden = false;
  /** @type {"new" | "rename"} */
  let projectFormMode = "new";
  // Type-ahead in the menu: what's been typed, and when.
  let typed = "";
  let typedAt = 0;
  const projectNewForm = /** @type {HTMLFormElement | null} */ (document.getElementById("project-new-form"));
  const projectNewName = /** @type {HTMLInputElement | null} */ (document.getElementById("project-new-name"));
  const projectWillBe = document.getElementById("project-will-be");
  const projectError = document.getElementById("project-error");
  const headerProject = document.getElementById("header-project");
  const heroWhere = document.getElementById("hero-where");
  const heroProject = document.getElementById("hero-project");
  const statProject = document.getElementById("stat-project");

  /** A file's path within its project, falling back to its workspace path.
   * @param {{ name?: string, path: string }} file */
  function shownPath(file) {
    return file.name || file.path;
  }

  /** @param {string | null} project */
  function projectLabel(project) {
    return project ? `${project}/` : "workspace/";
  }

  function projectQuery() {
    return currentProject ? `?project=${encodeURIComponent(currentProject)}` : "";
  }

  /** The folder on disk of the project in view, once /api/projects has said. */
  function projectFolder() {
    if (!currentProject) return projectsInfo.root;
    return projectsInfo.projects.find((p) => p.name === currentProject)?.folder || "";
  }

  function renderProject() {
    const label = projectLabel(currentProject);
    if (projectName) projectName.textContent = label;
    if (statProject) statProject.textContent = label;
    if (headerProject) {
      headerProject.textContent = label;
      headerProject.hidden = !currentProject;
    }
    if (heroWhere && heroProject) {
      heroProject.textContent = label;
      heroWhere.hidden = !currentProject || projectStatus(currentProject) === "missing";
    }
    renderProjectMenu();
  }

  /** Bring `project` into view, as opening one of its chats does.
   * @param {string | null} project */
  function showProject(project) {
    if (project === currentProject) return;
    currentProject = project;
    if (project) {
      localStorage.setItem(PROJECT_KEY, project);
    } else {
      localStorage.removeItem(PROJECT_KEY);
    }
    renderProject();
    void loadFiles();
  }

  async function loadProjects() {
    try {
      const res = await fetch("/api/projects");
      if (!res.ok) return;
      projectsInfo = await res.json();
    } catch {
      return;
    }
    if (currentProject && projectStatus(currentProject) === "unknown" && !sessionId) {
      showProject(null); // gone since it was last in view, chats and all
      void loadSessions();
      return;
    }
    renderProject();
    void loadFiles(); // the empty state shows the folder's path
  }

  /** @param {string} name @returns {"listed" | "hidden" | "missing" | "unknown"} */
  function projectStatus(name) {
    const found = projectsInfo.projects.find((p) => p.name === name);
    if (found) return found.hidden ? "hidden" : "listed";
    return projectsInfo.missing.includes(name) ? "missing" : "unknown";
  }

  /** One row of the menu: a project, with a line under its name.
   * @param {string | null} name @param {string} sub @param {() => void} pick */
  function projectItem(name, sub, pick) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "project-item";
    btn.dataset.project = name ?? "";
    btn.dataset.label = projectLabel(name);
    const current = name === currentProject;
    if (current) btn.setAttribute("aria-current", "true");
    btn.innerHTML = current ? '<i class="ph ph-check" aria-hidden="true"></i>' : "<span></span>";
    const label = document.createElement("span");
    label.className = "project-item-name";
    label.textContent = projectLabel(name);
    if (sub) {
      const line = document.createElement("span");
      line.className = "project-item-sub";
      line.textContent = sub;
      label.append(line);
    }
    // Under the name, so a narrow rail never cuts the name short.
    const state = document.createElement("span");
    state.className = "project-item-state";
    label.append(state);
    btn.append(label);
    btn.addEventListener("click", pick);
    li.append(btn);
    return li;
  }

  function renderProjectMenu() {
    if (!projectList) return;
    // Redrawn while open when the list arrives: keep keyboard focus in place.
    const focused = /** @type {HTMLElement | null} */ (projectList.querySelector(":focus"));
    const focusedProject = focused?.dataset.project;
    const rows = [projectItem(null, "Files outside any project", () => switchProject(null))];
    const hidden = projectsInfo.projects.filter((p) => p.hidden && p.name !== currentProject);
    for (const p of projectsInfo.projects) {
      if (p.hidden && p.name !== currentProject) continue;
      if (p.hidden) {
        rows.push(projectItem(p.name, "Hidden: picking it lists it again", () => void unhideProject(p.name)));
      } else {
        rows.push(projectItem(p.name, "", () => switchProject(p.name)));
      }
    }
    for (const name of projectsInfo.missing) {
      rows.push(projectItem(name, "Folder missing: its chats can still be read", () => switchProject(name)));
    }
    if (hidden.length && showHidden) {
      for (const p of hidden) {
        rows.push(projectItem(p.name, "Hidden: picking it lists it again", () => void unhideProject(p.name)));
      }
    } else if (hidden.length) {
      const li = document.createElement("li");
      const more = document.createElement("button");
      more.type = "button";
      more.className = "project-item project-action";
      more.innerHTML = '<i class="ph ph-eye" aria-hidden="true"></i>';
      const text = document.createElement("span");
      text.textContent = `Show ${hidden.length} hidden`;
      more.append(text);
      more.addEventListener("click", () => {
        showHidden = true;
        renderProjectMenu();
        /** @type {HTMLElement | null} */ (projectList.querySelector(`[data-project="${CSS.escape(hidden[0].name)}"]`))?.focus();
      });
      li.append(more);
      rows.push(li);
    }
    projectList.replaceChildren(...rows);
    for (const btn of projectList.querySelectorAll("[data-project]")) {
      if (/** @type {HTMLElement} */ (btn).dataset.project === focusedProject) /** @type {HTMLElement} */ (btn).focus();
    }
    // Rename and hide act on the project in view, while its folder is there.
    const manageable = currentProject !== null && ["listed", "hidden"].includes(projectStatus(currentProject));
    if (projectRenameBtn) projectRenameBtn.hidden = !manageable || !projectNewForm?.hidden;
    if (projectHideBtn) {
      projectHideBtn.hidden = !manageable || !projectNewForm?.hidden || projectStatus(currentProject ?? "") === "hidden";
    }
    paintActivity();
  }

  function openProjectMenu() {
    if (!projectMenu || !projectButton || !projectList) return;
    showHidden = false;
    renderProjectMenu();
    void loadProjects(); // folders may have been added on disk
    projectMenu.hidden = false;
    projectButton.setAttribute("aria-expanded", "true");
    const target = projectList.querySelector('[aria-current="true"]') || projectList.querySelector("button");
    /** @type {HTMLElement | null} */ (target)?.focus();
  }

  /** @param {{ focus?: boolean }} [opts] */
  function closeProjectMenu(opts = {}) {
    if (!projectMenu || !projectButton || projectMenu.hidden) return;
    projectMenu.hidden = true;
    projectButton.setAttribute("aria-expanded", "false");
    hideProjectForm();
    if (opts.focus) projectButton.focus();
  }

  /** Switch to `project` and start a new chat there. A chat still working
   * elsewhere carries on, and its project's rail shows it.
   * @param {string | null} project */
  function switchProject(project) {
    closeProjectMenu({ focus: false });
    if (project === currentProject) return;
    showProject(project);
    resetChat();
    question.focus();
  }

  /** The folder name the server will make from what's typed (app/projects.py).
   * @param {string} title */
  function folderNameFor(title) {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64)
      .replace(/-+$/, "");
  }

  /** @param {string} message */
  function showProjectError(message) {
    if (!projectError) return;
    projectError.textContent = message;
    projectError.hidden = false;
  }

  /** @param {string} name @param {{ name?: string, hidden?: boolean }} change */
  async function changeProject(name, change) {
    try {
      const res = await fetch(`/api/projects/${encodeURIComponent(name)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(change),
      });
      if (!res.ok) {
        showProjectError(await responseDetail(res));
        return null;
      }
      return /** @type {{ name: string }} */ (await res.json());
    } catch {
      showProjectError("Could not change the project. Is Humphrey still running?");
      return null;
    }
  }

  /** @param {string} name */
  async function unhideProject(name) {
    if (!(await changeProject(name, { hidden: false }))) return;
    await loadProjects();
    if (name === currentProject) {
      closeProjectMenu({ focus: true });
      showToast(`${projectLabel(name)} is listed again`);
    } else {
      switchProject(name);
    }
  }

  /** The New project form, also used to rename the project in view.
   * @param {"new" | "rename"} mode */
  function showProjectForm(mode) {
    if (!projectNewForm || !projectNewName) return;
    projectFormMode = mode;
    for (const btn of [projectNewBtn, projectRenameBtn, projectHideBtn]) {
      if (btn) btn.hidden = true;
    }
    projectNewForm.hidden = false;
    if (projectFormLabel) projectFormLabel.textContent = mode === "new" ? "Project name" : "New name";
    if (projectFormSubmit) projectFormSubmit.textContent = mode === "new" ? "Create" : "Rename";
    projectNewName.value = mode === "rename" ? currentProject ?? "" : "";
    if (projectWillBe) projectWillBe.textContent = "";
    if (projectError) projectError.hidden = true;
    projectNewName.focus();
    projectNewName.select();
  }

  function hideProjectForm() {
    if (!projectNewForm || !projectNewBtn) return;
    projectNewForm.hidden = true;
    projectNewBtn.hidden = false;
    if (projectError) projectError.hidden = true;
    renderProjectMenu(); // brings back Rename and Hide where they apply
  }

  projectButton?.addEventListener("click", () => {
    if (projectMenu?.hidden) {
      openProjectMenu();
    } else {
      closeProjectMenu({ focus: true });
    }
  });

  projectNewBtn?.addEventListener("click", () => showProjectForm("new"));
  projectRenameBtn?.addEventListener("click", () => showProjectForm("rename"));

  projectHideBtn?.addEventListener("click", async () => {
    const name = currentProject;
    if (!name || !(await changeProject(name, { hidden: true }))) return;
    await loadProjects();
    switchProject(null);
    showToast(`Hid ${projectLabel(name)}. Its folder and chats are kept; "Show hidden" lists it again.`);
  });

  projectNewName?.addEventListener("input", () => {
    const name = folderNameFor(projectNewName.value);
    const verb = projectFormMode === "new" ? "Creates" : "Renames it to";
    const unchanged = projectFormMode === "rename" && name === currentProject;
    if (projectWillBe) projectWillBe.textContent = name && !unchanged ? `${verb} projects/${name}/` : "";
    if (projectError) projectError.hidden = true;
  });

  document.getElementById("project-new-cancel")?.addEventListener("click", () => {
    hideProjectForm();
    projectNewBtn?.focus();
  });

  projectNewForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!projectNewName) return;
    if (projectFormMode === "rename") {
      const from = currentProject;
      if (!from) return;
      const renamed = await changeProject(from, { name: projectNewName.value });
      if (!renamed) return;
      await loadProjects();
      closeProjectMenu({ focus: false });
      showProject(renamed.name);
      void loadSessions();
      // The open chat's artefacts are served from the new folder now.
      if (sessionId) void openSession(sessionId, { silent: true });
      if (renamed.name !== from) showToast(`Renamed to ${projectLabel(renamed.name)}`);
      return;
    }
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: projectNewName.value }),
      });
      if (!res.ok) {
        showProjectError(await responseDetail(res));
        return;
      }
      const made = await res.json();
      await loadProjects();
      switchProject(made.name);
      showToast(`Made ${projectLabel(made.name)}`);
    } catch {
      showProjectError("Could not make the project. Is Humphrey still running?");
    }
  });

  projectMenu?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeProjectMenu({ focus: true });
      return;
    }
    if (event.target instanceof HTMLInputElement) return;
    const items = /** @type {HTMLElement[]} */ (
      [...projectMenu.querySelectorAll(".project-item")].filter((el) => !(/** @type {HTMLElement} */ (el).hidden))
    );
    // Typing jumps to the project whose name starts with what's typed.
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      typed = (Date.now() - typedAt < 800 ? typed : "") + event.key.toLowerCase();
      typedAt = Date.now();
      items.find((el) => (el.dataset.label || "").toLowerCase().startsWith(typed))?.focus();
      event.preventDefault();
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const at = items.indexOf(/** @type {HTMLElement} */ (document.activeElement));
    const step = event.key === "ArrowDown" ? 1 : -1;
    items[(at + step + items.length) % items.length]?.focus();
    event.preventDefault();
  });

  document.addEventListener("pointerdown", (event) => {
    const switcher = document.getElementById("project-switcher");
    if (switcher && event.target instanceof Node && !switcher.contains(event.target)) {
      closeProjectMenu({ focus: false });
    }
  });

  // Activity (/api/activity): which chats have a turn under way, in every
  // project. The chat this tab follows shows its own run; the rest come
  // from here, so the rail can mark a chat working or waiting on you
  // anywhere, including ones another tab started.
  const ACTIVITY_MS = 4000;
  const projectNeeds = document.getElementById("project-needs");
  const railNeeds = document.getElementById("rail-needs");
  /** @type {{ session_id: string, project: string | null, state: "working" | "needs_you" | "queued" }[]} */
  let activity = [];
  // When the list was last asked for: a task card drawn after that can't
  // be judged by it yet.
  let activityAt = 0;
  // Set while the page reopens the chat to follow a turn it didn't start.
  let catchingUp = false;

  async function loadActivity() {
    let next;
    const askedAt = Date.now();
    try {
      const res = await fetch("/api/activity");
      if (!res.ok) return;
      next = await res.json();
    } catch {
      return;
    }
    const ended = activity.some(
      (a) => a.project === currentProject && !next.some((n) => n.session_id === a.session_id),
    );
    const listed = (/** @type {string} */ id) =>
      Boolean(sessionList?.querySelector(`[data-session-id="${CSS.escape(id)}"]`));
    const unlisted = next.some((/** @type {any} */ a) => a.project === currentProject && !listed(a.session_id));
    activity = next;
    activityAt = askedAt;
    paintActivity();
    // The open chat started a turn this page isn't following: a task's
    // result came back, or another tab sent something. Follow it.
    const open = sessionId;
    if (open && !activeRun && !catchingUp && activity.some((a) => a.session_id === open && a.state !== "queued")) {
      catchingUp = true;
      void openSession(open, { silent: true }).finally(() => {
        catchingUp = false;
      });
    }
    // A chat that ended has a new title and time; a new one in its first
    // turn isn't in the list yet.
    if (ended || unlisted) void loadSessions();
  }

  /** @param {string | null} project */
  function projectState(project) {
    const mine = activity.filter((a) => a.project === project);
    if (mine.some((a) => a.state === "needs_you")) return "needs_you";
    return mine.length ? "working" : null;
  }

  /** @param {HTMLElement | null} badge @param {number} count @param {string} what */
  function setNeedsBadge(badge, count, what) {
    if (!badge) return;
    badge.hidden = count === 0;
    badge.textContent = String(count);
    badge.title = `${count} ${count === 1 ? "chat needs" : "chats need"} you ${what}`;
  }

  function paintActivity() {
    for (const item of document.querySelectorAll('#transcript .task-card[data-kind="started"]')) {
      const card = /** @type {HTMLElement} */ (item);
      if (activityAt < Number(card.dataset.since || 0)) continue;
      const state = activity.find((a) => a.session_id === card.dataset.taskId)?.state || "finished";
      card.dataset.state = state;
      const word = card.querySelector(".task-state");
      if (word) word.textContent = TASK_WORDS[state] || state;
    }
    for (const row of document.querySelectorAll("#session-list .session-item[data-session-id]")) {
      const el = /** @type {HTMLElement} */ (row);
      const state = activity.find((a) => a.session_id === el.dataset.sessionId)?.state;
      const meta = /** @type {HTMLElement | null} */ (el.querySelector(".session-meta"));
      if (state) {
        el.dataset.state = state;
      } else {
        delete el.dataset.state;
      }
      if (meta) {
        meta.textContent = state === "needs_you" ? "needs you" : state === "working" ? "working" : meta.dataset.time || "";
      }
    }
    for (const item of document.querySelectorAll("#project-list .project-item")) {
      const el = /** @type {HTMLElement} */ (item);
      const slot = /** @type {HTMLElement | null} */ (el.querySelector(".project-item-state"));
      if (!slot) continue;
      const state = projectState(el.dataset.project || null);
      slot.dataset.state = state || "";
      slot.textContent = state === "needs_you" ? "needs you" : state === "working" ? "working" : "";
    }
    const waiting = activity.filter((a) => a.state === "needs_you");
    setNeedsBadge(projectNeeds, waiting.filter((a) => a.project !== currentProject).length, "in other projects");
    // The collapsed rail hides every row, so it counts all but the open chat.
    setNeedsBadge(railNeeds, waiting.filter((a) => a.session_id !== sessionId).length, "elsewhere");
  }

  setInterval(() => {
    if (!document.hidden) void loadActivity();
  }, ACTIVITY_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) void loadActivity();
  });

  /** Mermaid theme from the live CSS tokens, so diagrams follow light/dark. */
  function mermaidTheme() {
    const css = getComputedStyle(document.documentElement);
    const token = (name) => css.getPropertyValue(name).trim();
    return {
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: {
        darkMode: currentTheme() === "dark",
        fontFamily: token("--font-sans"),
        background: token("--color-elevated"),
        primaryColor: token("--color-high"),
        primaryTextColor: token("--color-fg"),
        primaryBorderColor: token("--hb-accent"),
        lineColor: token("--color-muted"),
        edgeLabelBackground: token("--color-elevated"),
        textColor: token("--color-fg"),
        secondaryColor: token("--color-inset"),
        tertiaryColor: token("--color-elevated"),
        clusterBkg: token("--color-inset"),
        clusterBorder: token("--hb-accent"),
        actorBkg: token("--color-high"),
        actorBorder: token("--hb-accent"),
        actorTextColor: token("--color-fg"),
        signalColor: token("--color-fg-2"),
        signalTextColor: token("--color-fg"),
        noteBkgColor: token("--color-inset"),
        noteTextColor: token("--color-fg"),
        noteBorderColor: token("--color-border-strong"),
      },
    };
  }

  /**
   * @param {any} candidate
   * @returns {any | null}
   */
  function mermaidApi(candidate) {
    if (candidate && typeof candidate.initialize === "function" && typeof candidate.render === "function") {
      return candidate;
    }
    if (candidate && candidate.default) {
      return mermaidApi(candidate.default);
    }
    return null;
  }

  /** @param {string} src */
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`failed to load ${src}`));
      document.head.append(script);
    });
  }

  function resolveMermaid() {
    return (
      mermaidApi(window.mermaid) ||
      mermaidApi(window.__esbuild_esm_mermaid_nm && window.__esbuild_esm_mermaid_nm.mermaid)
    );
  }

  /** @returns {Promise<any | null>} */
  function loadMermaid() {
    if (mermaidLoader) {
      return mermaidLoader;
    }
    mermaidLoader = (async () => {
      let mermaid = resolveMermaid();
      if (!mermaid) {
        await loadScript(`/vendor/mermaid.min.js${ASSET_QUERY}`);
        mermaid = resolveMermaid();
      }
      if (!mermaid) {
        return null;
      }
      mermaid.initialize(mermaidTheme());
      return mermaid;
    })().catch(() => {
      mermaidLoader = null;
      return null;
    });
    return mermaidLoader;
  }

  /**
   * LLM mermaid often puts parentheses in unquoted labels. Mermaid treats `(` as a
   * node-shape token (PS), so those fences parse-fail while simpler diagrams render.
   * @param {string} source
   */
  function quoteMermaidLabels(source) {
    let out = source.replace(/\|([^|\n]+)\|/g, (full, inner) => {
      const trimmed = inner.trim();
      if (
        (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
        (trimmed.startsWith("'") && trimmed.endsWith("'"))
      ) {
        return full;
      }
      if (/[()[\]{}<>/]/.test(trimmed)) {
        return `|"${trimmed.replace(/"/g, "#quot;")}"|`;
      }
      return full;
    });
    out = out.replace(/(\b[A-Za-z][\w-]*)\[([^\]\n]+)\]/g, (full, id, inner) => {
      const trimmed = inner.trim();
      if (trimmed.startsWith('"') || trimmed.startsWith("'")) {
        return full;
      }
      if (/[()/]/.test(trimmed)) {
        return `${id}["${trimmed.replace(/"/g, "#quot;")}"]`;
      }
      return full;
    });
    return out.replace(/(-->|---|-.->)\s+end\b/g, '$1 mermaidEnd["end"]');
  }

  /** @param {unknown} err */
  function mermaidErrorMessage(err) {
    const raw = err && typeof err === "object" ? err.str || err.message : "";
    const line = String(raw || "")
      .split("\n")
      .map((row) => row.trim())
      .find(Boolean);
    return line ? line.slice(0, 180) : "";
  }

  /** @param {HTMLElement} root */
  /** Diagrams bake their colours in at render time, so redraw on theme change. */
  async function rethemeMermaid() {
    const rendered = [...document.querySelectorAll(".md-mermaid[data-rendered='1']")];
    if (!rendered.length || !mermaidLoader) {
      return;
    }
    const mermaid = await mermaidLoader;
    if (!mermaid) {
      return;
    }
    mermaid.initialize(mermaidTheme());
    for (const block of rendered) {
      const src = document.createElement("pre");
      src.className = "md-mermaid-src";
      src.textContent = block.dataset.source || "";
      block.replaceChildren(src);
      delete block.dataset.rendered;
    }
    await hydrateMermaid(document);
  }

  async function hydrateMermaid(root) {
    const blocks = [...root.querySelectorAll(".md-mermaid:not([data-rendered='1'])")];
    if (!blocks.length) {
      return;
    }
    const mermaid = await loadMermaid();
    if (!mermaid) {
      blocks.forEach((block) => {
        block.classList.add("is-fallback");
        if (!block.querySelector(".md-mermaid-error")) {
          const note = document.createElement("p");
          note.className = "md-mermaid-error";
          note.textContent = "Could not load diagram renderer.";
          block.prepend(note);
        }
      });
      return;
    }
    for (const block of blocks) {
      const srcEl = block.querySelector(".md-mermaid-src");
      const source = (srcEl?.textContent || "").trim();
      if (!source) {
        continue;
      }
      mermaidSeq += 1;
      const id = `hb-mmd-${mermaidSeq}`;
      const prepared = quoteMermaidLabels(source);
      try {
        const { svg } = await mermaid.render(id, prepared);
        const frame = document.createElement("div");
        frame.className = "md-mermaid-svg";
        frame.innerHTML = svg;
        const head = document.createElement("div");
        head.className = "md-mermaid-head";
        const label = document.createElement("span");
        label.textContent = "Diagram";
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "code-copy";
        copyBtn.textContent = "Copy";
        copyBtn.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(source);
            copyBtn.textContent = "Copied";
            setTimeout(() => {
              copyBtn.textContent = "Copy";
            }, 1200);
          } catch {
            copyBtn.textContent = "Failed";
          }
        });
        head.append(label, copyBtn);
        block.replaceChildren(head, frame);
        block.dataset.rendered = "1";
        block.dataset.source = source;
      } catch (err) {
        document.getElementById(id)?.remove();
        block.classList.add("is-error");
        if (!block.querySelector(".md-mermaid-error")) {
          const note = document.createElement("p");
          note.className = "md-mermaid-error";
          const detail = mermaidErrorMessage(err);
          note.textContent = detail
            ? `Could not render diagram: ${detail}`
            : "Could not render diagram.";
          block.prepend(note);
        }
        console.warn("mermaid render failed", err);
      }
    }
  }

  /** @param {HTMLElement} root */
  function lastTextNode(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.textContent) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (parent && parent.closest(".code-block-head, .code-copy, button")) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    let last = null;
    let current = walker.nextNode();
    while (current) {
      last = current;
      current = walker.nextNode();
    }
    return last;
  }

  /** @param {HTMLElement} root */
  function attachStreamCaret(root) {
    root.querySelectorAll(".stream-caret").forEach((node) => node.remove());
    const caret = document.createElement("span");
    caret.className = "stream-caret";
    caret.setAttribute("aria-hidden", "true");
    const text = lastTextNode(root);
    if (text) {
      text.after(caret);
      return;
    }
    root.append(caret);
  }

  /** @param {HTMLElement} root */
  function clearStreamCaret(root) {
    root.querySelectorAll(".stream-caret").forEach((node) => node.remove());
  }

  /** @param {HTMLElement} root */
  function bindCodeCopy(root) {
    root.querySelectorAll(".code-copy").forEach((btn) => {
      if (!(btn instanceof HTMLButtonElement) || btn.dataset.bound === "1") {
        return;
      }
      btn.dataset.bound = "1";
      btn.addEventListener("click", async () => {
        const block = btn.closest(".code-block");
        const code = block?.querySelector("code")?.textContent ?? "";
        try {
          await navigator.clipboard.writeText(code);
          btn.textContent = "Copied";
          setTimeout(() => {
            btn.textContent = "Copy";
          }, 1200);
        } catch {
          btn.textContent = "Failed";
        }
      });
    });
  }

  /**
   * @param {number} n
   * @returns {string}
   */
  function compactNum(n) {
    const abs = Math.abs(n);
    if (abs >= 1_000_000) {
      return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
    }
    if (abs >= 10_000) {
      return `${Math.round(n / 1000)}k`;
    }
    if (abs >= 1000) {
      return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
    }
    return String(n);
  }

  /**
   * @param {string | undefined} iso
   * @returns {string}
   */
  function relativeTime(iso) {
    if (!iso) {
      return "";
    }
    const then = Date.parse(iso);
    if (!Number.isFinite(then)) {
      return "";
    }
    const sec = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (sec < 45) {
      return "just now";
    }
    if (sec < 3600) {
      return `${Math.max(1, Math.round(sec / 60))}m ago`;
    }
    if (sec < 86400) {
      return `${Math.max(1, Math.round(sec / 3600))}h ago`;
    }
    if (sec < 604800) {
      return `${Math.max(1, Math.round(sec / 86400))}d ago`;
    }
    return new Date(then).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }

  function resetUsage() {
    tokensIn = 0;
    tokensOut = 0;
    cachedIn = 0;
    reasoningOut = 0;
    contextTokens = 0;
  }

  function updateStats() {
    const turns = messages.filter((m) => m.role === "user").length;
    // Without any reported usage (nothing sent yet), estimate from text.
    const reported = tokensIn > 0 || tokensOut > 0;
    const shownIn = reported ? tokensIn : Math.round(charsIn / 4);
    const shownOut = reported ? tokensOut : Math.round(charsOut / 4);
    const inContext = contextTokens || (reported ? 0 : shownIn + shownOut);
    const pct = Math.min(100, Math.round((inContext / contextBudget) * 100));

    if (statTurns) {
      statTurns.textContent = String(turns);
    }
    if (statTools) {
      statTools.textContent = String(toolsUsed);
    }
    if (statIo) {
      statIo.textContent = `${compactNum(shownIn)} in · ${compactNum(shownOut)} out`;
      statIo.title = reported
        ? `This chat so far, all model calls: ${tokensIn.toLocaleString()} in ` +
          `(${cachedIn.toLocaleString()} from cache), ${tokensOut.toLocaleString()} out ` +
          `(${reasoningOut.toLocaleString()} reasoning)`
        : "Estimated from the text; no model usage reported yet";
    }
    if (statContext) {
      statContext.title = `${inContext.toLocaleString()} of ${contextBudget.toLocaleString()} tokens`;
    }
    if (statContext) {
      statContext.textContent = `${pct}%`;
    }
    stage.dataset.context = pct >= 90 ? "hot" : pct >= 70 ? "warn" : "ok";
    if (contextBarFill) {
      contextBarFill.style.width = `${pct}%`;
    }
    if (statsRingFill) {
      const filled = (pct / 100) * RING_CIRC;
      statsRingFill.setAttribute("stroke-dasharray", `${filled} ${RING_CIRC}`);
    }
    if (statsRingPct) {
      statsRingPct.textContent = String(pct);
    }
    if (statsToggle) {
      statsToggle.title = `Details · context ${pct}%`;
    }
  }

  /**
   * @param {"user" | "assistant" | "error" | "status" | "context-summary"} kind
   * @param {string} content
   * @param {{ animate?: boolean, streaming?: boolean, images?: string[] }} [opts]
   */
  function appendMessage(kind, content, opts = {}) {
    if (kind === "context-summary") {
      return appendContextSummary(content, opts);
    }

    const wrap = document.createElement("article");
    wrap.className = `msg msg-${kind}`;
    if (opts.streaming) {
      wrap.classList.add("is-streaming");
    }
    if (opts.animate !== false) {
      wrap.classList.add("msg-enter");
    }

    if (kind === "assistant") {
      const role = document.createElement("div");
      role.className = "msg-role";
      role.textContent = "Humphrey";
      wrap.append(role);
    }

    // Skipped only for an image-only user message (no caption) - streaming
    // assistant replies rely on this element always existing to append into
    // later (see state.assistantEl.querySelector(".msg-body")), even before
    // any text has arrived.
    if (content || kind !== "user") {
      const body = document.createElement("div");
      body.className = "msg-body";
      if (kind === "assistant") {
        setMarkdown(body, content);
      } else {
        body.textContent = content;
      }
      wrap.append(body);
    }

    // Local preview URLs for a message just sent; for a reopened chat, the
    // server's image URLs, which never change, so the browser caches them.
    if (kind === "user" && opts.images && opts.images.length > 0) {
      const gallery = document.createElement("div");
      gallery.className = "msg-user-images";
      for (const url of opts.images) {
        const img = document.createElement("img");
        img.src = url;
        img.loading = "lazy";
        img.decoding = "async";
        // Its height is unknown until it loads; stay at the bottom if pinned.
        img.addEventListener("load", pinToBottom, { once: true });
        img.alt = "Attached image";
        gallery.append(img);
      }
      wrap.append(gallery);
    }

    if (kind === "assistant" && !opts.streaming) {
      wrap.append(buildMsgActions(content));
    }

    transcript.append(wrap);
    pinToBottom();
    return wrap;
  }

  /**
   * A compacted-history summary (app/context_window.py) is not something the
   * user typed, so it's rendered as a collapsed, clearly-labelled block with
   * real markdown formatting rather than the plain-text user bubble - full
   * detail stays reachable (expand it), just not dominating the transcript.
   * @param {string} content
   * @param {{ animate?: boolean }} [opts]
   */
  function appendContextSummary(content, opts = {}) {
    const details = document.createElement("details");
    details.className = "msg msg-context-summary";
    if (opts.animate !== false) {
      details.classList.add("msg-enter");
    }

    const summary = document.createElement("summary");
    summary.className = "context-summary-label";
    summary.textContent = "Context summary (earlier messages compacted)";
    details.append(summary);

    const body = document.createElement("div");
    body.className = "msg-body";
    setMarkdown(body, content);
    details.append(body);

    transcript.append(details);
    pinToBottom();
    return details;
  }

  /** @param {string} content */
  function buildMsgActions(content) {
    const actions = document.createElement("div");
    actions.className = "msg-actions";

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "msg-action";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(content);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1200);
      } catch {
        copyBtn.textContent = "Failed";
      }
    });

    actions.append(copyBtn);
    return actions;
  }

  /**
   * @param {number} ms
   * @returns {string}
   */
  function formatThoughtMs(ms) {
    if (ms < 1000) {
      return `${Math.max(1, Math.round(ms))}ms`;
    }
    return `${(ms / 1000).toFixed(1).replace(/\.0$/, "")}s`;
  }

  /**
   * @param {{
   *   thoughtEl: HTMLDetailsElement | null,
   *   thoughtStarted: number,
   * }} state
   */
  function ensureThought(state) {
    if (state.thoughtEl && state.thoughtEl.isConnected) {
      return state.thoughtEl;
    }
    const details = document.createElement("details");
    details.className = "thought msg-enter is-live";
    details.open = true;
    state.thoughtStarted = performance.now();

    const summary = document.createElement("summary");
    summary.className = "thought-summary";
    const breath = document.createElement("span");
    breath.className = "thought-breath";
    const label = document.createElement("span");
    label.className = "thought-label";
    label.textContent = "Thinking";
    breath.append(label);
    const meta = document.createElement("span");
    meta.className = "thought-meta";
    summary.append(breath, meta);

    const body = document.createElement("div");
    body.className = "thought-body";
    body.textContent = "Working…";

    details.append(summary, body);
    transcript.append(details);
    state.thoughtEl = details;
    startThoughtTimer(state);
    pinToBottom();
    return details;
  }

  /**
   * @param {{ thoughtEl: HTMLDetailsElement | null, thoughtStarted: number }} state
   */
  function startThoughtTimer(state) {
    stopThoughtTimer();
    let wordIndex = 0;
    let lastWordAt = performance.now();
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    thoughtTimer = window.setInterval(() => {
      if (!state.thoughtEl || !state.thoughtEl.isConnected) {
        stopThoughtTimer();
        return;
      }
      const now = performance.now();
      const meta = state.thoughtEl.querySelector(".thought-meta");
      if (meta) {
        meta.textContent = formatThoughtMs(now - state.thoughtStarted);
      }
      if (
        !reduceMotion &&
        state.thoughtEl.classList.contains("is-live") &&
        now - lastWordAt >= THOUGHT_WORD_MS
      ) {
        wordIndex = (wordIndex + 1) % THOUGHT_WORDS.length;
        lastWordAt = now;
        const label = state.thoughtEl.querySelector(".thought-label");
        if (label) {
          label.textContent = THOUGHT_WORDS[wordIndex];
        }
      }
    }, 100);
  }

  function stopThoughtTimer() {
    if (thoughtTimer) {
      window.clearInterval(thoughtTimer);
      thoughtTimer = null;
    }
  }

  /**
   * @param {{ thoughtEl: HTMLDetailsElement | null, thoughtStarted: number }} state
   * @param {string} [status]
   * @param {{ markdown?: boolean }} [opts]
   */
  function updateThoughtBody(state, status, opts = {}) {
    const body = state.thoughtEl?.querySelector(".thought-body");
    if (!body || !status) {
      return;
    }
    if (opts.markdown) {
      setMarkdown(body, status, { caret: true });
    } else {
      body.textContent = status;
    }
  }

  /**
   * @param {{ thoughtEl: HTMLDetailsElement | null, thoughtStarted: number }} state
   */
  function settleThought(state) {
    stopThoughtTimer();
    const el = state.thoughtEl;
    if (!el || !el.isConnected) {
      state.thoughtEl = null;
      return;
    }
    if (!el.classList.contains("is-live")) {
      return;
    }
    const ms = performance.now() - state.thoughtStarted;
    const usedTools = (el.querySelector(".thought-body")?.textContent || "").startsWith("Using");
    const hasThinking = Boolean(state.thinkingText);
    if (ms < 120 && !usedTools && !hasThinking) {
      el.remove();
      state.thoughtEl = null;
      return;
    }
    el.open = false;
    el.classList.remove("is-live");
    const label = el.querySelector(".thought-label");
    const meta = el.querySelector(".thought-meta");
    if (label) {
      label.textContent = "Thought";
    }
    if (meta) {
      meta.textContent = formatThoughtMs(ms);
    }
    const body = el.querySelector(".thought-body");
    if (body && (body.textContent || "").trim() === "Working…") {
      body.textContent = "Done.";
    } else if (body instanceof HTMLElement) {
      // Real thinking may have streamed in an earlier turn than the one that
      // triggered settling (state.thinkingText resets every message_start),
      // so check the DOM directly rather than trusting that flag here.
      clearStreamCaret(body);
    }
  }

  function showError(text) {
    pinnedToBottom = true;
    updateLatestButton();
    const el = appendMessage("error", text);
    // A ChatGPT plan's usage limit: where to go next is the main action.
    if (el && text.includes(MANAGE_USAGE_URL)) {
      const link = document.createElement("a");
      link.className = "btn-primary error-action";
      link.href = MANAGE_USAGE_URL;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Manage usage";
      el.append(link);
    }
    announce(`Error: ${text}`);
  }

  /** @param {unknown} value */
  function formatJson(value) {
    try {
      return JSON.stringify(value ?? {}, null, 2);
    } catch {
      return String(value);
    }
  }

  /** @param {string} raw */
  function prettyResult(raw) {
    try {
      return JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
      return raw;
    }
  }

  /** @param {string} patch */
  function diffStats(patch) {
    let added = 0;
    let removed = 0;
    for (const line of patch.split("\n")) {
      if (line.startsWith("+") && !line.startsWith("+++")) added += 1;
      else if (line.startsWith("-") && !line.startsWith("---")) removed += 1;
    }
    return { added, removed };
  }

  /**
   * One short phrase for the card meta.
   * @param {string} tool
   * @param {string} raw
   * @param {{ diff?: unknown, exit_code?: unknown, is_error?: unknown }} event
   */
  function summarizeToolResult(tool, raw, event) {
    if (tool === "edit" && typeof event.diff === "string") {
      const { added, removed } = diffStats(event.diff);
      return `+${added} −${removed}`;
    }
    if (tool === "bash" && typeof event.exit_code === "number") {
      return `exit ${event.exit_code}`;
    }
    if (tool === "read" && event.is_error !== true) {
      const lines = raw.split("\n").length;
      return `${lines} line${lines === 1 ? "" : "s"}`;
    }
    if (tool === "write" && event.is_error !== true) {
      return "written";
    }
    const flat = raw.replace(/\s+/g, " ").trim();
    return flat.length > 72 ? `${flat.slice(0, 69)}…` : flat || "done";
  }

  /**
   * Unified diff as coloured lines (built with textContent, never innerHTML).
   * @param {string} patch
   */
  function renderDiff(patch) {
    const pre = document.createElement("pre");
    pre.className = "tool-output tool-diff";
    for (const line of patch.replace(/\n$/, "").split("\n")) {
      const row = document.createElement("span");
      row.className = line.startsWith("@@")
        ? "diff-hunk"
        : line.startsWith("+++") || line.startsWith("---")
          ? "diff-file"
          : line.startsWith("+")
            ? "diff-add"
            : line.startsWith("-")
              ? "diff-del"
              : "diff-ctx";
      row.textContent = `${line}\n`;
      pre.append(row);
    }
    return pre;
  }

  /**
   * @param {unknown} args
   * @param {string} display
   */
  function toolSignature(display, args) {
    if (!args || typeof args !== "object") {
      return `${display}()`;
    }
    const entries = Object.entries(/** @type {Record<string, unknown>} */ (args));
    if (!entries.length) {
      return `${display}()`;
    }
    const parts = entries.slice(0, 3).map(([k, v]) => {
      const raw = typeof v === "string" ? v : JSON.stringify(v);
      const short = raw && raw.length > 28 ? `${raw.slice(0, 25)}…` : raw;
      return `${k}=${short}`;
    });
    const more = entries.length > 3 ? ", …" : "";
    return `${display}(${parts.join(", ")}${more})`;
  }

  /** Keep Details' "Active" line on the latest running tool, or mark it done. */
  function refreshActiveTool() {
    const running = [...liveTools.values()].filter((t) => t.el.dataset.state === "running");
    if (!running.length) {
      if (lastToolSummary.endsWith(" · running")) {
        setToolStat(lastToolSummary.replace(/ · running$/, " · done"));
      }
      return;
    }
    setToolStat(`${running[running.length - 1].display} · running`);
  }

  /**
   * @param {{
   *   toolGroupEl: HTMLElement | null,
   *   toolGroupCount: number,
   *   turn: number | null
   * }} state
   * @param {number | null} turn
   */
  function ensureToolGroup(state, turn) {
    if (state.toolGroupEl && state.toolGroupEl.isConnected) {
      return state.toolGroupEl;
    }
    const group = document.createElement("div");
    group.className = "tool-group msg-enter";
    group.dataset.count = "0";
    if (turn != null) {
      group.dataset.turn = String(turn);
    }
    const label = document.createElement("span");
    label.className = "tool-group-label";
    label.textContent = "Tools";
    group.append(label);
    transcript.append(group);
    state.toolGroupEl = group;
    state.toolGroupCount = 0;
    return group;
  }

  /**
   * Repeat calls to the same tool within one turn (e.g. read
   * across several files) collapse into one cluster
   * instead of stacking up as separate panels - the thing that made a busy
   * turn feel overwhelming. A lone call is unaffected: the cluster's own
   * summary line stays hidden until a second call of the same tool shows up,
   * so a single tool call still looks exactly like a normal row.
   * @param {{ toolGroupEl: HTMLElement | null, toolGroupCount: number, toolClusters: Map<string, any>, turn: number | null }} state
   * @param {string} tool
   */
  function ensureToolCluster(state, tool) {
    if (!state.toolClusters) {
      state.toolClusters = new Map();
    }
    const existing = state.toolClusters.get(tool);
    if (existing && existing.details.isConnected) {
      return existing;
    }

    const details = document.createElement("details");
    details.className = "tool-cluster";
    details.open = true; // a lone call renders exactly like a plain row
    details.dataset.state = "running";
    details.dataset.running = "1"; // drives the band; see updateToolClusterChrome
    details.dataset.userToggled = "0";

    const summary = document.createElement("summary");
    summary.className = "tool-cluster-summary";
    summary.hidden = true; // revealed once this becomes an actual cluster (2+)
    summary.addEventListener("click", () => {
      details.dataset.userToggled = "1";
    });

    const dot = document.createElement("span");
    dot.className = "tool-status-dot";
    dot.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    label.className = "tool-cluster-label";
    const meta = document.createElement("span");
    meta.className = "tool-cluster-meta";
    summary.append(dot, label, meta);

    const body = document.createElement("div");
    body.className = "tool-cluster-body";
    details.append(summary, body);

    ensureToolGroup(state, state.turn).append(details);

    const cluster = { details, summary, label, meta, body, tool, total: 0, running: 0, done: 0, error: 0 };
    state.toolClusters.set(tool, cluster);
    return cluster;
  }

  /** @param {{ details: HTMLDetailsElement, summary: HTMLElement, label: HTMLElement, meta: HTMLElement, tool: string, total: number, running: number, done: number, error: number }} cluster */
  function updateToolClusterChrome(cluster) {
    // Apart from state: a cluster with a failure can still have calls running.
    cluster.details.dataset.running = cluster.running > 0 ? "1" : "0";
    const isCluster = cluster.total > 1;
    cluster.summary.hidden = !isCluster;
    if (!isCluster) {
      return;
    }
    cluster.label.textContent = `${cluster.tool} ×${cluster.total}`;
    const userToggled = cluster.details.dataset.userToggled === "1";
    if (cluster.error > 0) {
      cluster.details.dataset.state = "error";
      cluster.meta.textContent =
        cluster.running > 0
          ? `${cluster.running} running · ${cluster.error} error${cluster.error === 1 ? "" : "s"}`
          : `${cluster.error} error${cluster.error === 1 ? "" : "s"}`;
      cluster.details.open = true; // never let a collapsed cluster hide a failure
      if (cluster.running === 0) {
        stopDotPulse(cluster.summary);
      }
    } else if (cluster.running > 0) {
      cluster.details.dataset.state = "running";
      cluster.meta.textContent = `${cluster.running} running`;
      if (!userToggled) {
        cluster.details.open = false;
      }
    } else {
      cluster.details.dataset.state = "done";
      cluster.meta.textContent = "done";
      stopDotPulse(cluster.summary);
      if (!userToggled) {
        cluster.details.open = false;
      }
    }
  }

  /**
   * @param {string} tool
   * @param {string} callId
   * @param {unknown} args
   * @param {string} [label]
   * @param {{
   *   toolGroupEl: HTMLElement | null,
   *   toolGroupCount: number,
   *   turn: number | null
   * }} [groupState]
   */
  function appendToolRow(tool, callId, args, label, groupState) {
    toolSeq += 1;
    toolsUsed += 1;
    updateStats();
    const panelId = `tool-panel-${toolSeq}`;
    const display = label || tool;
    // A templated label ("Reading {path}") already names its arguments.
    const signature = label && label !== tool ? label : toolSignature(tool, args);

    const row = document.createElement("div");
    row.className = "tool msg-enter";
    row.dataset.callId = callId;
    row.dataset.state = "running";
    row.dataset.userToggled = "0";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "tool-toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-controls", panelId);
    toggle.setAttribute("aria-label", `${display}, running. Activate to show details.`);

    const dot = document.createElement("span");
    dot.className = "tool-status-dot";
    dot.setAttribute("aria-hidden", "true");

    const name = document.createElement("span");
    name.className = "tool-name";
    name.textContent = signature;

    const meta = document.createElement("span");
    meta.className = "tool-meta";
    meta.textContent = "running";

    toggle.append(dot, name, meta);

    const panel = document.createElement("div");
    panel.className = "tool-panel";
    panel.id = panelId;

    const argsLabel = document.createElement("span");
    argsLabel.className = "tool-section-label";
    argsLabel.textContent = "Args";

    const argsPre = document.createElement("pre");
    argsPre.className = "tool-output";
    const argsText = formatJson(args);
    const hasArgs = argsText !== "{}" && argsText !== "null";
    argsPre.textContent = hasArgs ? argsText : "{}";

    const resultLabel = document.createElement("span");
    resultLabel.className = "tool-section-label";
    resultLabel.textContent = "Result";

    const output = document.createElement("pre");
    output.className = "tool-output";
    output.textContent = "…";

    // Always show args for transparency; empty object still reads as intentional.
    panel.append(argsLabel, argsPre, resultLabel, output);
    panel.hidden = true; // collapsed by default - the running dot is enough while it's live

    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") === "true";
      row.dataset.userToggled = "1";
      setToolExpanded({ toggle, panel, el: row }, !open);
      if (!open) {
        pinToBottom();
      }
    });

    row.append(toggle, panel);

    let cluster = null;
    if (groupState) {
      const group = ensureToolGroup(groupState, groupState.turn);
      groupState.toolGroupCount += 1;
      group.dataset.count = String(groupState.toolGroupCount);
      const groupLabel = group.querySelector(".tool-group-label");
      if (groupLabel) {
        const n = groupState.toolGroupCount;
        groupLabel.textContent = n === 1 ? "Tool" : `${n} tools`;
      }
      cluster = ensureToolCluster(groupState, tool);
      cluster.total += 1;
      cluster.running += 1;
      cluster.body.append(row);
      updateToolClusterChrome(cluster);
    } else {
      transcript.append(row);
    }

    pinToBottom();
    setToolStat(`${display} · running`);

    const card = {
      el: row,
      toggle,
      name,
      meta,
      output,
      panel,
      display,
      signature,
      preview: hasArgs ? argsText.replace(/\s+/g, " ").slice(0, 48) : "",
      argsText: hasArgs ? argsText : "",
      started: performance.now(),
    };
    liveTools.set(callId, card);
    refreshActiveTool();
    return card;
  }

  /**
   * Settle a tool card with its outcome: the live stream's tool_end, or a
   * tool row from a reopened chat (which has no duration, so ``ms`` is null).
   * Returns whether the run failed.
   * @param {any} card from appendToolRow
   * @param {string} tool
   * @param {string} rawResult
   * @param {{ is_error?: unknown, diff?: unknown, exit_code?: unknown, question?: any }} outcome
   * @param {{ toolClusters?: Map<string, any> }} state
   * @param {number | null} ms
   */
  function finishToolCard(card, tool, rawResult, outcome, state, ms) {
    const blurb = summarizeToolResult(tool, rawResult, outcome);
    const isError = outcome.is_error === true;
    const took = ms == null ? "" : ` · ${ms} ms`;
    // Edits and shell runs carry their outcome on the row itself.
    card.meta.textContent =
      tool === "edit" || tool === "bash" ? `${blurb}${took}` : ms == null ? (isError ? "error" : "done") : `${ms} ms`;
    if (tool === "ask_user_question" && outcome.question) {
      const words = { answered: "answered", skipped: "skipped", replied: "replied", stopped: "no answer" };
      card.meta.textContent = words[/** @type {keyof typeof words} */ (outcome.question.status)] || "done";
    }
    if (typeof outcome.diff === "string" && outcome.diff) {
      const diffLabel = document.createElement("span");
      diffLabel.className = "tool-section-label";
      diffLabel.textContent = "Changes";
      (card.output.previousElementSibling || card.output).before(diffLabel, renderDiff(outcome.diff));
    }
    card.el.dataset.state = isError ? "error" : "done";
    stopDotPulse(card.el);
    card.preview = blurb;
    card.toggle.setAttribute(
      "aria-label",
      ms == null
        ? `${card.display}, ${blurb}. Activate to show full output.`
        : `${card.display}, ${blurb}, ${ms} milliseconds. Activate to show full output.`,
    );
    card.output.textContent = prettyResult(rawResult);

    if (card.el.dataset.userToggled !== "1") {
      setToolExpanded(card, isError);
    }
    const cluster = state.toolClusters?.get(tool);
    if (cluster) {
      cluster.running = Math.max(0, cluster.running - 1);
      if (isError) {
        cluster.error += 1;
      } else {
        cluster.done += 1;
      }
      updateToolClusterChrome(cluster);
    }
    return isError;
  }

  /**
   * @param {{ toggle: HTMLButtonElement, panel: HTMLElement, el: HTMLElement }} card
   * @param {boolean} open
   */
  function setToolExpanded(card, open) {
    card.toggle.setAttribute("aria-expanded", open ? "true" : "false");
    card.panel.hidden = !open;
    card.el.classList.toggle("is-open", open);
  }

  /**
   * Live-verified against a real (slow, error-returning) tool call: once the
   * live-pulse box-shadow keyframes have actually painted a frame, Chromium
   * does not tear the CSSAnimation down just because data-state stops
   * matching `[data-state="running"]` in the cascade - getComputedStyle and
   * getAnimations() both keep reporting it as playing indefinitely, so the
   * dot goes on visibly pulsing long after the row reads "done"/"error".
   * Cancelling it explicitly is the only reliable fix. No-op for a dot whose
   * animation never actually started (fast calls that resolved before a
   * paint), and safe on browsers without getAnimations().
   * @param {HTMLElement} container
   */
  function stopDotPulse(container) {
    const dot = container.querySelector(".tool-status-dot");
    if (!dot || !dot.getAnimations) {
      return;
    }
    for (const anim of dot.getAnimations()) {
      anim.cancel();
    }
  }

  /**
   * A run can end (stop button, network drop, server error) while a tool call
   * is still in flight, so it never gets the tool_end that would move it off
   * data-state="running" - left alone, its dot pulses forever even
   * though the run itself is long over. Called once per run, from the
   * sendPrompt finally block, to settle anything still marked running.
   * @param {{ toolEls: Map<string, any>, toolClusters: Map<string, any> }} state
   */
  function finalizeStuckTools(state) {
    for (const card of state.toolEls.values()) {
      if (card.el.dataset.state === "running" || card.el.dataset.state === "waiting") {
        card.el.closest(".tool-cluster")?.removeAttribute("data-waiting");
        card.el.dataset.state = "stopped";
        card.meta.textContent = "stopped";
        card.toggle.setAttribute(
          "aria-label",
          `${card.display}, stopped before finishing. Activate to show details.`,
        );
        stopDotPulse(card.el);
      }
    }
    for (const cluster of state.toolClusters.values()) {
      if (cluster.running > 0) {
        cluster.running = 0;
        cluster.details.dataset.state = "stopped";
        cluster.meta.textContent = "stopped";
        stopDotPulse(cluster.summary);
      }
    }
  }

  // --- Question cards (ask_user_question) -----------------------------------
  // The model asks; the app always adds "Something else" and Skip. One pick
  // answers a single question. The card waits on the run (`pendingAsk`) until
  // the tool ends, then settles to a short record of the answer.

  /** @type {any} */
  let pendingAsk = null;
  let askSeq = 0;
  const baseTitle = document.title;
  const composerPlaceholder = question.getAttribute("placeholder") || "";

  /** @param {string} cls @param {string} text */
  function askButton(cls, text) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = cls;
    btn.textContent = text;
    return btn;
  }

  /**
   * @param {number} key
   * @param {string} label
   * @param {string} description
   * @param {boolean} recommended
   * @param {boolean} multi
   * @param {boolean} [other]
   */
  function askOption(key, label, description, recommended, multi, other = false) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `ask-option${other ? " is-other" : ""}`;
    btn.setAttribute("role", multi ? "checkbox" : "radio");
    btn.setAttribute("aria-checked", "false");
    const keyEl = document.createElement("span");
    keyEl.className = "ask-key";
    keyEl.setAttribute("aria-hidden", "true");
    keyEl.textContent = String(key);
    const text = document.createElement("span");
    const labelEl = document.createElement("span");
    labelEl.className = "ask-label";
    labelEl.append(label);
    if (recommended) {
      const tag = document.createElement("span");
      tag.className = "ask-tag";
      tag.textContent = "Recommended";
      labelEl.append(tag);
    }
    text.append(labelEl);
    if (description) {
      const desc = document.createElement("span");
      desc.className = "ask-desc";
      desc.textContent = description;
      text.append(desc);
    }
    btn.append(keyEl, text);
    return btn;
  }

  /**
   * One answer as words: the choices, then any typed text.
   * @param {{ choices?: string[], text?: string | null, skipped?: boolean }} answer
   */
  function askAnswerWords(answer) {
    if (!answer || answer.skipped) {
      return "Skipped";
    }
    const parts = [];
    if (answer.choices && answer.choices.length) {
      parts.push(answer.choices.join(", "));
    }
    if (answer.text) {
      parts.push(`“${answer.text}”`);
    }
    return parts.join("; ") || "Skipped";
  }

  /**
   * A question card. `onSubmit` sends the answer body and says if it landed.
   * @param {any[]} questions
   * @param {{ onSubmit: (body: any) => Promise<boolean>, note?: string, offline?: boolean }} opts
   */
  function createAskCard(questions, opts) {
    askSeq += 1;
    const id = askSeq;
    const card = document.createElement("section");
    card.className = "ask-card msg-enter";
    card.dataset.state = opts.offline ? "offline" : "waiting";
    card.tabIndex = -1;
    card.setAttribute("aria-label", "Humphrey is asking you");

    const head = document.createElement("div");
    head.className = "ask-head";
    const status = document.createElement("span");
    status.className = "ask-status";
    const dot = document.createElement("span");
    dot.className = "ask-dot";
    dot.setAttribute("aria-hidden", "true");
    const statusText = document.createElement("span");
    statusText.textContent = opts.note || "Needs you";
    status.append(dot, statusText);
    head.append(status);
    const steps = document.createElement("div");
    steps.className = "ask-steps";
    if (questions.length > 1) {
      head.append(steps);
    }

    const body = document.createElement("div");
    body.className = "ask-body";
    const foot = document.createElement("div");
    foot.className = "ask-foot";
    const skipBtn = askButton("btn-ghost ask-skip", "Skip");
    const hint = document.createElement("span");
    hint.className = "ask-hint";
    const sendBtn = askButton("btn-primary ask-send", "Send");
    foot.append(skipBtn, hint, sendBtn);
    card.append(head, body, foot);

    /** @type {{ choices: Set<string>, text: string, skipped: boolean, done: boolean }[]} */
    const answers = questions.map(() => ({ choices: new Set(), text: "", skipped: false, done: false }));
    let step = 0;
    let busy = false;
    /** @type {HTMLButtonElement[]} */
    let optionEls = [];
    /** @type {HTMLButtonElement | null} */
    let otherBtn = null;
    /** @type {HTMLTextAreaElement | null} */
    let otherField = null;
    const chips = questions.map((q, i) => {
      const chip = askButton("ask-step", q.header);
      chip.addEventListener("click", () => show(i, true));
      steps.append(chip);
      return chip;
    });

    const last = () => step === questions.length - 1;

    function syncChecks() {
      const q = questions[step];
      const a = answers[step];
      q.options.forEach((/** @type {any} */ opt, /** @type {number} */ k) => {
        optionEls[k].setAttribute("aria-checked", a.choices.has(opt.label) ? "true" : "false");
      });
      if (otherBtn) {
        otherBtn.setAttribute("aria-checked", a.text.trim() ? "true" : "false");
      }
    }

    function syncFoot() {
      const q = questions[step];
      const a = answers[step];
      const has = a.choices.size > 0 || a.text.trim() !== "";
      sendBtn.textContent = last() ? "Send" : "Next";
      sendBtn.hidden = !(q.multiSelect || a.text.trim() !== "" || (questions.length > 1 && has));
      sendBtn.disabled = !has;
      const n = q.options.length + 1;
      hint.textContent = q.multiSelect
        ? `1–${n} to choose · Enter ${last() ? "to send" : "for next"}`
        : `1–${n} to pick`;
      chips.forEach((chip, j) => {
        chip.setAttribute("aria-current", j === step ? "step" : "false");
        chip.dataset.done = answers[j].done ? "1" : "0";
      });
    }

    /** @param {number} i @param {boolean} [focus] */
    function show(i, focus = false) {
      step = i;
      const q = questions[i];
      const a = answers[i];
      body.replaceChildren();
      const title = document.createElement("h3");
      title.className = "ask-question";
      title.id = `ask-${id}-q${i}`;
      title.textContent = q.question;
      const list = document.createElement("div");
      list.className = "ask-options";
      list.setAttribute("role", q.multiSelect ? "group" : "radiogroup");
      list.setAttribute("aria-labelledby", title.id);
      optionEls = q.options.map((/** @type {any} */ opt, /** @type {number} */ k) => {
        const el = askOption(k + 1, opt.label, opt.description || "", Boolean(opt.recommended), q.multiSelect);
        el.addEventListener("click", () => pick(opt.label));
        return el;
      });
      otherBtn = askOption(q.options.length + 1, "Something else…", "", false, q.multiSelect, true);
      otherField = document.createElement("textarea");
      otherField.className = "ask-other-field";
      otherField.rows = 2;
      otherField.placeholder = "Type your answer";
      otherField.setAttribute("aria-label", `Your own answer to: ${q.question}`);
      otherField.value = a.text;
      otherField.hidden = a.text === "";
      otherBtn.addEventListener("click", () => {
        if (!otherField) return;
        otherField.hidden = false;
        otherField.focus();
      });
      otherField.addEventListener("input", () => {
        if (!otherField) return;
        a.text = otherField.value;
        if (!q.multiSelect && a.text.trim()) {
          a.choices.clear();
        }
        a.skipped = false;
        a.done = a.choices.size > 0 || a.text.trim() !== "";
        syncChecks();
        syncFoot();
      });
      otherField.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
          e.preventDefault();
          if (a.text.trim()) {
            advance();
          }
        }
      });
      list.append(...optionEls, otherBtn, otherField);
      body.append(title, list);
      syncChecks();
      syncFoot();
      if (focus) {
        (optionEls.find((el) => el.getAttribute("aria-checked") === "true") || optionEls[0])?.focus();
      }
    }

    /** @param {string} label */
    function pick(label) {
      if (busy) return;
      const q = questions[step];
      const a = answers[step];
      a.skipped = false;
      if (q.multiSelect) {
        if (a.choices.has(label)) a.choices.delete(label);
        else a.choices.add(label);
        a.done = a.choices.size > 0 || a.text.trim() !== "";
        syncChecks();
        syncFoot();
        return;
      }
      a.choices = new Set([label]);
      a.text = "";
      if (otherField) {
        otherField.value = "";
        otherField.hidden = true;
      }
      a.done = true;
      syncChecks();
      advance();
    }

    function advance() {
      const a = answers[step];
      a.done = a.skipped || a.choices.size > 0 || a.text.trim() !== "";
      const next = answers.findIndex((ans, j) => j > step && !ans.done);
      const firstOpen = answers.findIndex((ans) => !ans.done);
      if (next >= 0) {
        show(next, true);
      } else if (firstOpen >= 0) {
        show(firstOpen, true);
      } else {
        void submit();
      }
    }

    function skip() {
      if (busy) return;
      const a = answers[step];
      a.choices.clear();
      a.text = "";
      a.skipped = true;
      a.done = true;
      advance();
    }

    function answerBody() {
      if (answers.every((a) => a.skipped)) {
        return { skipped: true };
      }
      /** @type {Record<string, any>} */
      const out = {};
      questions.forEach((q, i) => {
        const a = answers[i];
        out[q.header] = a.skipped
          ? { skipped: true }
          : { choices: [...a.choices], text: a.text.trim() || null };
      });
      return { answers: out };
    }

    /** @param {any} [override] */
    async function submit(override) {
      if (busy) return false;
      busy = true;
      const before = card.dataset.state;
      card.dataset.state = "sending";
      const ok = await opts.onSubmit(override || answerBody());
      busy = false;
      if (!ok && card.dataset.state === "sending") {
        card.dataset.state = before || "waiting";
      }
      return ok;
    }

    skipBtn.addEventListener("click", () => skip());
    sendBtn.addEventListener("click", () => advance());
    card.addEventListener("keydown", (e) => {
      const target = /** @type {HTMLElement} */ (e.target);
      if (e.key === "Escape") {
        e.preventDefault();
        question.focus();
        return;
      }
      if (target instanceof HTMLTextAreaElement) return;
      const all = [...optionEls, ...(otherBtn ? [otherBtn] : [])];
      if (/^[1-9]$/.test(e.key) && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const el = all[Number(e.key) - 1];
        if (el) {
          e.preventDefault();
          el.click();
        }
        return;
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const at = all.indexOf(/** @type {HTMLButtonElement} */ (target));
        const to = at < 0 ? 0 : (at + (e.key === "ArrowDown" ? 1 : all.length - 1)) % all.length;
        e.preventDefault();
        all[to]?.focus();
        return;
      }
      if (e.key === "Enter" && questions[step].multiSelect && target.classList.contains("ask-option")) {
        e.preventDefault();
        if (answers[step].done) advance();
      }
    });

    show(0);

    return {
      el: card,
      questions,
      focus() {
        card.focus({ preventScroll: true });
        optionEls[0]?.focus({ preventScroll: true });
      },
      /** @param {string} text */
      reply(text) {
        return submit({ reply: text });
      },
    };
  }

  /**
   * Settle a card into the record of its answer (`details` is the tool's).
   * @param {HTMLElement} card
   * @param {any} details
   */
  function settleAskCard(card, details) {
    const statusWords = {
      answered: "You answered",
      skipped: "You skipped this",
      replied: "You replied",
      stopped: "Stopped before you answered",
    };
    const state = String(details.status || "answered");
    card.dataset.state = state;
    card.replaceChildren();
    const head = document.createElement("div");
    head.className = "ask-head";
    const status = document.createElement("span");
    status.className = "ask-status";
    const dot = document.createElement("span");
    dot.className = "ask-dot";
    dot.setAttribute("aria-hidden", "true");
    status.append(dot, statusWords[/** @type {keyof typeof statusWords} */ (state)] || "Answered");
    head.append(status);
    const list = document.createElement("dl");
    list.className = "ask-summary";
    const questions = Array.isArray(details.questions) ? details.questions : [];
    const answers = details.answers || {};
    // Each question stays with its answer, so a reopened chat still says
    // what was asked. Long questions are clamped to two lines (full text on hover).
    for (const q of questions) {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      const header = document.createElement("span");
      header.className = "ask-summary-header";
      header.textContent = q.header;
      const asked = document.createElement("span");
      asked.className = "ask-summary-question";
      asked.textContent = q.question;
      asked.title = q.question;
      dt.append(header, asked);
      const dd = document.createElement("dd");
      dd.textContent =
        state === "replied" ? "See your reply" : state === "stopped" ? "No answer" : askAnswerWords(answers[q.header]);
      row.append(dt, dd);
      list.append(row);
    }
    if (state === "replied") {
      const row = document.createElement("div");
      row.className = "ask-summary-reply";
      const dt = document.createElement("dt");
      dt.textContent = "Your reply";
      const dd = document.createElement("dd");
      dd.textContent = `“${details.reply || ""}”`;
      row.append(dt, dd);
      list.append(row);
    }
    card.append(head, list);
    card.setAttribute(
      "aria-label",
      `${statusWords[/** @type {keyof typeof statusWords} */ (state)] || "Answered"}: ${questions.map((/** @type {any} */ q) => q.question).join(" ")}`,
    );
  }

  /**
   * A card whose run has ended: answering sends the answer as your next
   * message, so a reload or a stop loses nothing.
   * @param {any[]} questions
   */
  function createOfflineAskCard(questions) {
    /** @type {any} */
    let ask = null;
    ask = createAskCard(questions, {
      offline: true,
      note: "Humphrey stopped waiting · your answer starts a new message",
      onSubmit: async (body) => {
        if (body.skipped) {
          settleAskCard(ask.el, { status: "skipped", questions, answers: {} });
          return true;
        }
        const lines = questions.map((q) => `${q.header}: ${askAnswerWords(body.answers?.[q.header])}`);
        settleAskCard(ask.el, { status: "answered", questions, answers: body.answers || {} });
        void sendPrompt(lines.join("\n"));
        return true;
      },
    });
    return ask;
  }

  /** @param {any} details @param {boolean} [superseded] */
  function renderStoredAsk(details, superseded = false) {
    const questions = Array.isArray(details?.questions) ? details.questions : [];
    if (details?.status === "stopped" && !superseded) {
      const ask = createOfflineAskCard(questions);
      ask.el.classList.remove("msg-enter");
      transcript.append(ask.el);
      return ask.el;
    }
    const card = document.createElement("section");
    card.className = "ask-card is-stored";
    settleAskCard(card, details);
    transcript.append(card);
    return card;
  }

  /** @param {boolean} waiting */
  function setAskWaiting(waiting) {
    document.title = waiting ? `Needs you · ${baseTitle}` : baseTitle;
    const running = stage.dataset.running === "1";
    setStatus(waiting ? "waiting" : running ? "working" : "ready");
    question.placeholder = waiting
      ? "Or answer in your own words…"
      : running
        ? "Steer the current turn…"
        : composerPlaceholder;
    if (composerHint && waiting) {
      composerHint.innerHTML = "<kbd>Enter</kbd> answer · <kbd>Esc</kbd> stop";
    }
  }

  /**
   * @param {string} sid
   * @param {string} requestId
   * @param {any} body
   */
  async function postAskAnswer(sid, requestId, body) {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sid)}/ui/${encodeURIComponent(requestId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        return true;
      }
      showToast(res.status === 404 ? "Humphrey isn't waiting for this any more" : "Could not send your answer");
    } catch {
      showToast("Could not send your answer");
    }
    return false;
  }

  // Background tasks (app/tasks.py). Humphrey starts one with start_task:
  // a card under that call follows the task, and a card where its result
  // came back shows the result. Both open the task's own chat.
  const TASK_WORDS = /** @type {Record<string, string>} */ ({
    working: "working",
    needs_you: "needs you",
    queued: "queued",
    finished: "finished",
    completed: "done",
    failed: "failed",
    stopped: "stopped",
  });

  /**
   * @param {"started" | "result"} kind
   * @param {string} taskId
   * @param {string} title
   * @param {string} state
   * @param {string} label
   */
  function taskCard(kind, taskId, title, state, label) {
    const card = document.createElement("section");
    card.className = "task-card msg-enter";
    card.dataset.kind = kind;
    card.dataset.taskId = taskId;
    card.dataset.state = state;
    card.dataset.since = String(Date.now());
    const head = document.createElement("div");
    head.className = "task-head";
    const what = document.createElement("span");
    what.textContent = label;
    const stateEl = document.createElement("span");
    stateEl.className = "task-state";
    stateEl.textContent = TASK_WORDS[state] || state;
    head.append(what, stateEl);
    const name = document.createElement("p");
    name.className = "task-title";
    name.textContent = title;
    const body = document.createElement("div");
    body.className = "task-body";
    const foot = document.createElement("div");
    foot.className = "task-foot";
    const open = document.createElement("button");
    open.type = "button";
    open.className = "btn-ghost btn-sm";
    open.textContent = "Open task";
    open.addEventListener("click", () => void openSession(taskId));
    foot.append(open);
    card.append(head, name, body, foot);
    return card;
  }

  /** The card under a start_task call, following its task.
   * @param {{ task_id: string, description: string }} task @param {{ animate?: boolean }} [opts] */
  function appendTaskStarted(task, opts = {}) {
    const card = taskCard("started", String(task.task_id), String(task.description || "Background task"), "working", "Background task");
    if (opts.animate === false) {
      card.classList.remove("msg-enter");
      card.dataset.since = "0"; // from history: the list already speaks for it
    }
    transcript.append(card);
    paintActivity();
    return card;
  }

  /** The card where a task's result came back.
   * @param {any} notice @param {{ animate?: boolean }} [opts] */
  function appendTaskResult(notice, opts = {}) {
    const status = String(notice.status || "completed");
    const label =
      status === "completed" ? "Background task finished" : status === "failed" ? "Background task failed" : "Background task stopped";
    const card = taskCard("result", String(notice.task_id || ""), String(notice.description || "Background task"), status, label);
    if (opts.animate === false) card.classList.remove("msg-enter");
    const body = /** @type {HTMLElement} */ (card.querySelector(".task-body"));
    if (notice.pending) {
      body.textContent = "The result was too long to include here, so Humphrey is reading it in full.";
    } else if (notice.result) {
      setMarkdown(body, String(notice.result));
    }
    transcript.append(card);
    return card;
  }

  /** A task's own chat names the chat that started it. @param {string} parentId */
  function appendTaskOrigin(parentId) {
    const row = document.createElement("div");
    row.className = "task-origin";
    const text = document.createElement("span");
    text.textContent = "A background task, started from another chat.";
    const back = document.createElement("button");
    back.type = "button";
    back.className = "link-btn";
    back.textContent = "Back to that chat";
    back.addEventListener("click", () => void openSession(parentId));
    row.append(text, back);
    transcript.append(row);
  }

  /**
   * The run is asking: show the card below the tool call and wait.
   * @param {Record<string, any>} event
   * @param {any} state
   */
  function showAsk(event, state) {
    const callId = String(event.call_id ?? "");
    const requestId = String(event.request_id ?? "");
    const questions = Array.isArray(event.questions) ? event.questions : [];
    const sid = sessionId;
    if (!sid || !requestId || questions.length === 0) {
      return;
    }
    const row = state.toolEls.get(callId);
    if (row) {
      row.el.dataset.state = "waiting";
      row.meta.textContent = "needs you";
      stopDotPulse(row.el);
      row.el.closest(".tool-cluster")?.setAttribute("data-waiting", "1");
    }
    const ask = createAskCard(questions, { onSubmit: (body) => postAskAnswer(sid, requestId, body) });
    transcript.append(ask.el);
    // Later tools start a fresh group below the card, in the order things happened.
    state.toolGroupEl = null;
    pendingAsk = { ...ask, callId };
    setAskWaiting(true);
    announce(`Humphrey is asking: ${questions[0].question}`);
    pinToBottom();
    ask.focus();
  }

  /** Forget the waiting card without touching it: its chat is going away. */
  function dropAsk() {
    pendingAsk = null;
    setAskWaiting(false);
  }

  /** @param {any} [details] */
  function endAsk(details) {
    if (!pendingAsk) return;
    const ask = pendingAsk;
    pendingAsk = null;
    setAskWaiting(false);
    if (details) {
      settleAskCard(ask.el, details);
    } else {
      ask.el.replaceWith(createOfflineAskCard(ask.questions).el);
    }
  }

  function renderQueue() {
    queueEl.replaceChildren();
    for (let i = 0; i < queue.length; i += 1) {
      const item = queue[i];
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `queue-chip${item.kind === "follow-up" ? " is-followup" : ""}`;
      chip.setAttribute("role", "listitem");
      const kind = document.createElement("span");
      kind.className = "queue-chip-kind";
      kind.textContent = item.kind === "follow-up" ? "Follow-up" : "Steer";
      const text = document.createElement("span");
      text.className = "queue-chip-text";
      text.textContent = item.text;
      chip.append(kind, text);
      chip.title = "Click to remove";
      chip.addEventListener("click", () => {
        queue.splice(i, 1);
        renderQueue();
      });
      queueEl.append(chip);
    }
  }

  function enqueueFollowUp(text) {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    queue.push({ kind: "follow-up", text: trimmed });
    question.value = "";
    autoSize();
    renderQueue();
    showToast("Follow-up queued");
  }

  async function drainQueue() {
    if (!queue.length || activeRun) {
      return;
    }
    const next = queue.shift();
    renderQueue();
    if (next) {
      await sendPrompt(next.text, { fromQueue: true });
    }
  }

  function resetChat() {
    dropAsk();
    leaveRun();
    messages = [];
    sessionId = null;
    localStorage.removeItem(SESSION_KEY);
    queue = [];
    renderQueue();
    liveTools.clear();
    refreshActiveTool();
    stopThoughtTimer();
    pinnedToBottom = true;
    hasNewBelow = false;
    toolsUsed = 0;
    charsIn = 0;
    charsOut = 0;
    resetUsage();
    clearTranscript();
    setActive(false);
    setRunning(false);
    setToolStat("—");
    updateStats();
    updateLatestButton();
    if (srAnnounce) {
      srAnnounce.textContent = "";
    }
    question.value = "";
    autoSize();
    thread.scrollTop = 0;
    void loadSessions();
    question.focus();
  }

  async function loadSessions() {
    if (!sessionList) {
      return;
    }
    const project = currentProject;
    try {
      const res = await fetch(`/api/sessions${projectQuery()}`);
      if (!res.ok) {
        return;
      }
      /** @type {{ id: string, title?: string, message_count?: number, updated_at?: string }[]} */
      const rows = await res.json();
      if (project !== currentProject) return; // a switch overtook this answer
      sessionList.replaceChildren();
      if (sessionsCount) {
        if (rows.length) {
          sessionsCount.hidden = false;
          sessionsCount.textContent = String(rows.length);
        } else {
          sessionsCount.hidden = true;
        }
      }
      if (!rows.length) {
        const empty = document.createElement("li");
        empty.className = "session-empty";
        empty.textContent = "No chats yet";
        sessionList.append(empty);
        return;
      }
      // A chat's background tasks go under it; one whose chat isn't listed
      // stands on its own.
      const listedIds = new Set(rows.map((/** @type {any} */ r) => r.id));
      /** @type {Map<string, any[]>} */
      const tasksOf = new Map();
      const top = [];
      for (const row of rows) {
        if (row.parent_id && listedIds.has(row.parent_id)) {
          tasksOf.set(row.parent_id, [...(tasksOf.get(row.parent_id) || []), row]);
        } else {
          top.push(row);
        }
      }
      for (const row of top.flatMap((r) => [r, ...(tasksOf.get(r.id) || [])])) {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = row.parent_id ? "session-item is-task" : "session-item";
        if (row.id === sessionId) {
          btn.classList.add("is-active");
        }
        const title = document.createElement("span");
        title.className = "session-title";
        title.textContent = row.title || "Untitled chat";
        const meta = document.createElement("span");
        meta.className = "session-meta";
        meta.textContent = relativeTime(row.updated_at) || "Saved";
        meta.dataset.time = meta.textContent;
        btn.dataset.sessionId = row.id;
        btn.append(title, meta);
        btn.addEventListener("click", () => {
          void openSession(row.id);
        });
        li.append(btn);
        sessionList.append(li);
      }
      paintActivity();
    } catch {
      /* ignore */
    }
  }

  /**
   * @param {string} id
   * @param {{ silent?: boolean, reopened?: boolean }} [opts]
   */
  async function openSession(id, opts = {}) {
    dropAsk();
    leaveRun();
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
      if (!res.ok) {
        if (sessionId === id) {
          sessionId = null;
          localStorage.removeItem(SESSION_KEY);
        }
        if (!opts.silent) {
          showToast("Could not open session");
        }
        return;
      }
      const data = await res.json();
      showProject(typeof data.project === "string" ? data.project : null);
      const parentId = typeof data.parent_id === "string" ? data.parent_id : null;
      const rows = (data.messages || []).filter(
        (m) => m && (m.content || m.role === "tool" || m.role === "task" || (Array.isArray(m.images) && m.images.length > 0)),
      );
      // `messages` counts the user/assistant turns shown in the UI; a
      // context-summary row is rendered below but kept out of it.
      messages = rows.filter((m) => m.role === "user" || m.role === "assistant");
      sessionId = id;
      localStorage.setItem(SESSION_KEY, id);
      clearTranscript();
      if (parentId) appendTaskOrigin(parentId);
      liveTools.clear();
      refreshActiveTool();
      toolsUsed = 0;
      charsIn = 0;
      charsOut = 0;
      // The server adds up every model call in the chat (usage_totals); the
      // latest call alone (last_usage) says how full the context is. The
      // loop below only fills the text-based estimate for chats with no
      // usage yet.
      resetUsage();
      const totals = data.usage_totals || {};
      tokensIn = Number(totals.input) || 0;
      tokensOut = Number(totals.output) || 0;
      cachedIn = Number(totals.cached_input) || 0;
      reasoningOut = Number(totals.reasoning_output) || 0;
      if (data.last_usage && typeof data.last_usage === "object") {
        contextTokens = (Number(data.last_usage.input) || 0) + (Number(data.last_usage.output) || 0);
      }
      setToolStat("—");
      /** @type {{ ts: number, el: Element }[]} */
      const placed = [];
      // A model step's tool runs share one group, as they did live.
      /** @type {any} */
      let toolGroup = null;
      for (const [i, msg] of rows.entries()) {
        let el;
        if (msg.role !== "tool" || (toolGroup && msg.tool && msg.tool.step !== toolGroup.step)) {
          toolGroup = null;
        }
        if (msg.role === "tool" && msg.tool) {
          const t = msg.tool;
          toolGroup = toolGroup || {
            toolEls: new Map(),
            toolGroupEl: null,
            toolGroupCount: 0,
            toolClusters: new Map(),
            turn: null,
            step: t.step,
          };
          const name = String(t.name || "tool");
          const callId = String(t.call_id || crypto.randomUUID());
          const firstInGroup = toolGroup.toolGroupCount === 0;
          const card = appendToolRow(name, callId, t.args || {}, typeof t.label === "string" ? t.label : name, toolGroup);
          finishToolCard(card, name, String(msg.content || ""), t, toolGroup, null);
          liveTools.delete(callId);
          // Artefacts are placed before a top-level row, so a group is placed
          // once, by its first run: a file a run made goes after its group.
          el = firstInGroup ? toolGroup.toolGroupEl : null;
          if (t.task) {
            appendTaskStarted(t.task, { animate: false });
            toolGroup = null;
          }
        } else if (msg.role === "task" && msg.task) {
          el = appendTaskResult(msg.task, { animate: false });
        } else if (msg.role === "context_summary") {
          el = appendMessage("context-summary", msg.content, { animate: false });
        } else if (msg.role === "question") {
          // Only a last, unanswered card stays answerable.
          const later = rows.slice(i + 1).some((r) => r.role === "user");
          el = renderStoredAsk(msg.question, later);
        } else {
          if (msg.role === "user") {
            charsIn += msg.content.length;
          } else {
            charsOut += msg.content.length;
          }
          el = appendMessage(msg.role, msg.content, {
            animate: false,
            images: Array.isArray(msg.images) ? msg.images : [],
          });
        }
        if (el && typeof msg.ts === "number") placed.push({ ts: msg.ts, el });
      }
      // Artefacts go back where they happened: before the first message
      // newer than them (so after the request, before the reply).
      const artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
      for (const art of [...artifacts].sort((a, b) => (a.created_ms ?? 0) - (b.created_ms ?? 0))) {
        const next = placed.find((row) => row.ts > (art.created_ms ?? 0));
        placeArtifact(art, { animate: false, before: next ? next.el : null });
      }
      // A turn still running here: the history stops before it, so show
      // what was typed to start it, and below, follow the run itself.
      const running = data.run && typeof data.run.prompt === "string" ? data.run : null;
      if (running && running.task) {
        appendTaskResult(running.task, { animate: false }); // a task reporting back started it
      } else if (running) {
        messages.push({ role: "user", content: running.prompt });
        appendMessage("user", running.prompt, { animate: false });
      }
      setToolStat("—"); // drawing past tool runs isn't activity
      setActive(messages.length > 0);
      setRunning(false);
      updateStats();
      pinnedToBottom = true;
      hasNewBelow = false;
      updateLatestButton();
      pinToBottom();
      void loadSessions();
      question.focus();
      if (running) {
        void showRun(
          (signal) =>
            fetch(`/api/sessions/${encodeURIComponent(id)}/stream`, {
              headers: { Accept: "text/event-stream" },
              signal,
            }),
          // It ended between the two requests, so its turn is stored now.
          async () => {
            if (!opts.reopened) void openSession(id, { silent: true, reopened: true });
          },
        );
      }
    } catch {
      if (sessionId === id) {
        sessionId = null;
        localStorage.removeItem(SESSION_KEY);
      }
      if (!opts.silent) {
        showToast("Could not open session");
      }
    }
  }

  async function restoreSession() {
    if (!sessionId) {
      return;
    }
    await openSession(sessionId, { silent: true });
  }

  async function probeHealth() {
    setStatus("connecting");
    try {
      const res = await fetch("/healthz");
      if (!res.ok) {
        throw new Error("health failed");
      }
      const data = await res.json();
      if (Number(data.context_window_tokens) > 0) {
        fallbackBudget = Number(data.context_window_tokens);
        applyContextBudget();
      }
      const providerLabels = {
        mock: "mock",
        chatgpt: "ChatGPT plan",
        openai: "OpenAI",
        azure_openai: "Azure OpenAI",
        unconfigured: "unconfigured",
      };
      const provider =
        typeof data.provider === "string" && providerLabels[data.provider]
          ? providerLabels[data.provider]
          : data.mock_mode
            ? "mock"
            : "azure";
      await loadModels();
      await loadAuth(typeof data.provider === "string" ? data.provider : "");
      if (statProvider) {
        statProvider.textContent = provider;
      }
      // A page opened onto a running turn may already be showing it.
      setStatus(pendingAsk ? "waiting" : stage.dataset.running === "1" ? "working" : "ready");
    } catch {
      setStatus("ready");
      if (statProvider) {
        statProvider.textContent = "local";
      }
    }
  }

  /**
   * @param {ReadableStream<Uint8Array>} body
   * @param {(event: Record<string, unknown>) => void} onEvent
   * @param {AbortSignal} signal
   */
  async function readSse(body, onEvent, signal) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      if (signal.aborted) {
        await reader.cancel();
        return;
      }
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        for (const line of part.split("\n")) {
          if (!line.startsWith("data:")) {
            continue;
          }
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") {
            continue;
          }
          try {
            onEvent(JSON.parse(raw));
          } catch {
            showError(`Bad SSE payload: ${raw.slice(0, 120)}`);
          }
        }
      }
    }
  }

  /**
   * The reply being written is done: drop its caret and draw it as Markdown,
   * or remove it if nothing was written. At its end, or at a stop or error.
   * @param {any} state
   */
  function settleReply(state) {
    if (state.assistantEl) {
      state.assistantEl.classList.remove("is-streaming");
      const body = state.assistantEl.querySelector(".msg-body");
      if (body instanceof HTMLElement) {
        if (state.assistantText) {
          setMarkdown(body, state.assistantText);
        } else {
          clearStreamCaret(body);
        }
      }
      if (state.assistantText) {
        state.assistantEl.append(buildMsgActions(state.assistantText));
      }
    }
    if (state.assistantText) {
      state.finalAssistantText = state.assistantText;
      charsOut += state.assistantText.length;
      updateStats();
    } else if (state.assistantEl) {
      state.assistantEl.remove();
    }
    state.assistantEl = null;
    state.assistantText = "";
  }

  /**
   * @param {Record<string, unknown>} event
   * @param {{
   *   assistantEl: HTMLElement | null,
   *   assistantText: string,
   *   finalAssistantText: string,
   *   toolEls: Map<string, any>,
   *   toolGroupEl: HTMLElement | null,
   *   toolGroupCount: number,
   *   turn: number | null,
   *   thoughtEl: HTMLDetailsElement | null,
   *   thoughtStarted: number
   * }} state
   */
  function applyEvent(event, state) {
    const type = String(event.type ?? "");

    if (type === "artifact") {
      placeArtifact(event);
      // Later tools in this turn start a fresh group below the card, so the
      // transcript stays in the order things happened.
      state.toolGroupEl = null;
      void loadFiles();
      pinToBottom();
      return;
    }

    if (type === "task_notification") {
      appendTaskResult(event);
      state.toolGroupEl = null;
      pinToBottom();
      void loadActivity();
      return;
    }

    if (type === "ui_request" && event.kind === "question") {
      showAsk(event, state);
      void loadActivity();
      return;
    }

    if (type === "agent_start") {
      if (typeof event.session_id === "string" && event.session_id) {
        sessionId = event.session_id;
        localStorage.setItem(SESSION_KEY, sessionId);
      }
      setRunning(true);
      ensureThought(state);
      void loadActivity();
      return;
    }

    if (type === "turn_start") {
      setRunning(true);
      state.turn = typeof event.turn === "number" ? event.turn : null;
      state.toolGroupEl = null;
      state.toolGroupCount = 0;
      state.toolClusters = new Map();
      // The model's next step after tools gets its own live panel below
      // them; otherwise the only sign it's still working sits above the
      // tool cards, often scrolled out of view.
      if (state.thoughtEl && state.thoughtEl.isConnected && transcript.lastElementChild !== state.thoughtEl) {
        settleThought(state);
        state.thoughtEl = null;
        state.thinkingText = "";
      }
      ensureThought(state);
      return;
    }

    if (type === "turn_end") {
      state.toolGroupEl = null;
      state.toolGroupCount = 0;
      state.toolClusters = new Map();
      return;
    }

    if (type === "message_start") {
      state.assistantText = "";
      state.thinkingText = "";
      ensureThought(state);
      setRunning(true);
      return;
    }

    if (type === "thinking_delta" && typeof event.text === "string") {
      state.thinkingText += event.text;
      ensureThought(state);
      updateThoughtBody(state, state.thinkingText, { markdown: true });
      pinToBottom();
      return;
    }

    if (type === "message_delta" && typeof event.text === "string") {
      state.assistantText += event.text;
      settleThought(state);
      if (!state.assistantEl) {
        state.assistantEl = appendMessage("assistant", "", {
          animate: false,
          streaming: true,
        });
      }
      const body = state.assistantEl.querySelector(".msg-body");
      if (body instanceof HTMLElement) {
        setMarkdown(body, state.assistantText, { caret: true });
      }
      pinToBottom();
      return;
    }

    if (type === "message_end") {
      if (event.usage && typeof event.usage === "object") {
        // One model call. The chat's totals add every call up; the context
        // is this call alone, since its input already holds the whole
        // conversation (adding those up would count it again each call).
        const u = event.usage;
        tokensIn += Number(u.input) || 0;
        tokensOut += Number(u.output) || 0;
        cachedIn += Number(u.cached_input) || 0;
        reasoningOut += Number(u.reasoning_output) || 0;
        contextTokens = (Number(u.input) || 0) + (Number(u.output) || 0);
        updateStats();
      }
      settleReply(state);
      return;
    }

    if (type === "user_message" && typeof event.text === "string") {
      const text = event.text.trim();
      if (!text) {
        return;
      }
      // Avoid duplicating a user bubble we already rendered for steer/follow-up.
      const last = messages[messages.length - 1];
      if (!(last && last.role === "user" && last.content === text)) {
        messages.push({ role: "user", content: text });
        charsIn += text.length;
        appendMessage("user", text);
        updateStats();
      }
      return;
    }

    if (type === "agent_end") {
      if (state.finalAssistantText) {
        messages.push({ role: "assistant", content: state.finalAssistantText });
        announce(state.finalAssistantText);
        state.finalAssistantText = "";
      }
      settleThought(state);
      setRunning(false);
      void loadSessions();
      return;
    }

    if (type === "tool_start") {
      const callId = String(event.call_id ?? crypto.randomUUID());
      const tool = String(event.tool ?? "tool");
      const label = typeof event.label === "string" ? event.label : tool;
      if (typeof event.turn === "number") {
        state.turn = event.turn;
      }
      ensureThought(state);
      // Don't clobber real streamed reasoning with the cosmetic placeholder -
      // only reasoning models produce thinkingText, and it's worth more than
      // "Using <tool>…" once it's there.
      if (!state.thinkingText) {
        updateThoughtBody(state, tool === "ask_user_question" ? "Waiting for your answer…" : `Using ${label}…`);
      }
      const card = appendToolRow(tool, callId, event.args ?? {}, label, state);
      state.toolEls.set(callId, card);
      setRunning(true);
      return;
    }

    if (type === "tool_end") {
      const callId = String(event.call_id ?? "");
      if (pendingAsk && pendingAsk.callId === callId) {
        endAsk(event.question || { status: "stopped", questions: pendingAsk.questions });
      }
      state.toolEls.get(callId)?.el.closest(".tool-cluster")?.removeAttribute("data-waiting");
      const tool = String(event.tool ?? "tool");
      const label = typeof event.label === "string" ? event.label : tool;
      const rawResult = String(event.result ?? "");
      let card = callId ? state.toolEls.get(callId) : undefined;

      if (!card) {
        card = appendToolRow(tool, callId || crypto.randomUUID(), {}, label, state);
        if (callId) {
          state.toolEls.set(callId, card);
        }
      }

      // The "Using <label>…" placeholder set at tool_start (see below) would
      // otherwise sit frozen here, describing an action that's already
      // finished, while the thought panel's dot and rotating word keep
      // animating for the rest of the run - reads as stuck even though the
      // agent is legitimately still working on the next step.
      if (!state.thinkingText) {
        updateThoughtBody(state, `Used ${label}.`);
      }

      const ms = Math.max(1, Math.round(performance.now() - card.started));
      const isError = finishToolCard(card, tool, rawResult, event, state, ms);
      if (event.task && typeof event.task === "object") {
        appendTaskStarted(/** @type {any} */ (event.task));
        state.toolGroupEl = null; // later tools start a group below the card
        void loadActivity();
      }
      if (callId) {
        liveTools.delete(callId);
      }
      setToolStat(`${card.display} · ${isError ? "error" : "done"} · ${ms} ms`);
      refreshActiveTool();
      pinToBottom();
      return;
    }

    if (type === "error") {
      settleReply(state);
      settleThought(state);
      showError(String(event.detail ?? "Unknown agent error"));
      setRunning(false);
    }

    if (type === "stopped") {
      settleReply(state); // what was written before the stop stays
      settleThought(state);
      appendMessage("status", "Stopped.");
      announce("Run stopped.");
      setRunning(false);
    }
  }

  /**
   * @param {"steer" | "follow-up"} kind
   * @param {string} content
   */
  async function postQueue(kind, content) {
    const text = content.trim();
    if (!text || !sessionId) {
      return false;
    }
    const path =
      kind === "steer"
        ? `/api/sessions/${sessionId}/steer`
        : `/api/sessions/${sessionId}/follow-up`;
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text }),
      });
      if (!res.ok) {
        return false;
      }
      messages.push({ role: "user", content: text });
      charsIn += text.length;
      appendMessage("user", text);
      question.value = "";
      autoSize();
      updateStats();
      showToast(kind === "steer" ? "Steering queued" : "Follow-up queued");
      return true;
    } catch {
      return false;
    }
  }

  /**
   * @param {string} content
   * @param {{ fromQueue?: boolean, steer?: boolean }} [opts]
   */
  async function sendPrompt(content, opts = {}) {
    const text = content.trim();
    const attachments = pendingAttachments;
    if (!text && attachments.length === 0) {
      return;
    }

    // While a question card waits, the message box answers it.
    if (pendingAsk && text && attachments.length === 0 && !opts.fromQueue) {
      if (await pendingAsk.reply(text)) {
        question.value = "";
        autoSize();
      }
      return;
    }

    // Once agent_end has arrived the server has no run to queue onto (the
    // stream may still be open while it names the session), so only
    // post to the server queue mid-turn; otherwise queue locally below.
    if (activeRun && sessionId && !opts.fromQueue && stage.dataset.running === "1") {
      const kind = opts.steer ? "steer" : "follow-up";
      if (await postQueue(kind, text)) {
        return;
      }
    }

    if (activeRun && !opts.steer && !opts.fromQueue) {
      enqueueFollowUp(text);
      return;
    }

    leaveRun();
    const wasEmpty = messages.length === 0;
    /** @type {{ role: "user"; content: string; images?: { data: string }[] }} */
    const pushedMessage = { role: "user", content: text };
    if (attachments.length > 0) {
      pushedMessage.images = attachments.map((a) => ({ data: a.data }));
    }
    messages.push(pushedMessage);
    charsIn += text.length;
    question.value = "";
    pendingAttachments = [];
    renderAttachmentPreviews();
    autoSize();
    updateStats();

    if (wasEmpty) {
      setActive(true);
    }

    pinnedToBottom = true;
    hasNewBelow = false;
    updateLatestButton();
    appendMessage("user", text, { images: attachments.map((a) => a.previewUrl) });

    await showRun(
      (signal) =>
        fetch("/api/agent", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          // Only the new turn: the server rebuilds everything earlier from the
          // stored session, so `messages` here is UI state, never resent.
          body: JSON.stringify({
            message: {
              content: pushedMessage.content,
              ...(pushedMessage.images ? { images: pushedMessage.images } : {}),
            },
            session_id: sessionId,
            // Only a new chat uses this; an existing one keeps its own project.
            ...(currentProject ? { project: currentProject } : {}),
            ...(selectedModel ? { model: selectedModel } : {}),
            ...(selectedEffort ? { reasoning_effort: selectedEffort } : {}),
          }),
          signal,
        }).finally(() => {
          // The bytes have been sent; don't keep them in the UI's message list.
          delete pushedMessage.images;
        }),
      async (res) => {
        messages.pop();
        transcript.querySelector(".msg-user:last-of-type")?.remove();
        if (messages.length === 0) {
          setActive(false);
        }
        showError(await responseDetail(res));
      },
    );
  }

  /** @param {Response} res */
  async function responseDetail(res) {
    try {
      const data = await res.json();
      return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data, null, 2);
    } catch {
      return `HTTP ${res.status}`;
    }
  }

  /**
   * Show a run as it happens, until its stream ends: a turn this tab just
   * started, or one already running that it has come back to. The run
   * belongs to the server, so leaving only stops the showing (leaveRun);
   * Stop is what ends the run (stopRun).
   * @param {(signal: AbortSignal) => Promise<Response>} open
   * @param {(res: Response) => Promise<void>} onRefused
   */
  async function showRun(open, onRefused) {
    const controller = new AbortController();
    activeRun = controller;

    /** @type {{
     *   assistantEl: HTMLElement | null,
     *   assistantText: string,
     *   finalAssistantText: string,
     *   thinkingText: string,
     *   toolEls: Map<string, any>,
     *   toolGroupEl: HTMLElement | null,
     *   toolGroupCount: number,
     *   toolClusters: Map<string, any>,
     *   turn: number | null,
     *   thoughtEl: HTMLDetailsElement | null,
     *   thoughtStarted: number
     * }} */
    const state = {
      assistantEl: null,
      assistantText: "",
      finalAssistantText: "",
      thinkingText: "",
      toolEls: new Map(),
      toolGroupEl: null,
      toolGroupCount: 0,
      toolClusters: new Map(),
      turn: null,
      thoughtEl: null,
      thoughtStarted: 0,
    };

    setRunning(true);
    ensureThought(state);

    try {
      const res = await open(controller.signal);

      const headerSession = res.headers.get("X-Session-Id");
      if (headerSession) {
        sessionId = headerSession;
        localStorage.setItem(SESSION_KEY, sessionId);
      }

      if (!res.ok) {
        await onRefused(res);
        return;
      }

      if (!res.body) {
        showError("No response body from the run's stream");
        return;
      }

      await readSse(res.body, (ev) => applyEvent(ev, state), controller.signal);
    } catch (err) {
      if (err && typeof err === "object" && "name" in err && err.name === "AbortError") {
        return; // this tab left the run; it carries on
      }
      showError(String(err));
    } finally {
      if (activeRun === controller) {
        activeRun = null;
        endAsk();
        finalizeStuckTools(state);
        liveTools.clear();
        refreshActiveTool();
        settleThought(state);
        setRunning(false);
        void loadSessions();
        void loadActivity();
        question.focus();
        void drainQueue();
      }
    }
  }

  /** Stop showing the run here (another chat, a new one). It carries on. */
  function leaveRun() {
    if (activeRun) {
      activeRun.abort();
      activeRun = null;
    }
  }

  /** End the run itself. Its stream says so, then ends once the turn is saved. */
  async function stopRun() {
    if (!activeRun || !sessionId) {
      return;
    }
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      // 409: nothing left to stop, the turn is only being saved.
      if (res.ok || res.status === 409) {
        return;
      }
    } catch {
      /* fall through */
    }
    showToast("Could not stop the run");
  }

  // Rails
  document.getElementById("sessions-toggle")?.addEventListener("click", () => {
    setSessionsOpen(stage.dataset.sessionsOpen !== "1");
  });
  document.getElementById("sessions-collapse")?.addEventListener("click", () => {
    setSessionsOpen(false);
  });
  document.getElementById("stats-toggle")?.addEventListener("click", () => {
    setStatsOpen(stage.dataset.statsOpen !== "1");
  });
  document.getElementById("stats-collapse")?.addEventListener("click", () => {
    setStatsOpen(false);
  });
  detailsBtn?.addEventListener("click", () => {
    setStatsOpen(stage.dataset.statsOpen !== "1");
  });
  applyTheme(currentTheme());
  scrim?.addEventListener("click", () => {
    setSessionsOpen(false);
    setStatsOpen(false);
  });

  // Always start collapsed; users open rails deliberately each session.
  setSessionsOpen(false);
  setStatsOpen(false);

  thread.addEventListener("scroll", () => syncPinFromScroll(), { passive: true });
  latestBtn?.addEventListener("click", () => jumpToLatest());
  newBtn?.addEventListener("click", () => resetChat());
  newBtnCollapsed?.addEventListener("click", () => resetChat());
  stopBtn?.addEventListener("click", () => void stopRun());
  followUpBtn?.addEventListener("click", () => {
    if (stage.dataset.running === "1" && sessionId) {
      void postQueue("follow-up", question.value);
      return;
    }
    enqueueFollowUp(question.value);
  });
  toastEl?.addEventListener("click", () => {
    toastEl.hidden = true;
  });

  attachBtn?.addEventListener("click", () => attachInput?.click());
  attachInput?.addEventListener("change", () => {
    if (attachInput.files && attachInput.files.length > 0) {
      void addAttachments(attachInput.files);
    }
    attachInput.value = "";
  });
  question.addEventListener("paste", (event) => {
    const items = event.clipboardData?.items;
    if (!items) {
      return;
    }
    const imageFiles = Array.from(items)
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file) => file !== null);
    if (imageFiles.length > 0) {
      event.preventDefault();
      void addAttachments(imageFiles);
    }
  });

  question.addEventListener("input", autoSize);
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.altKey && stage.dataset.running === "1") {
      event.preventDefault();
      if (sessionId) {
        void postQueue("follow-up", question.value);
      } else {
        enqueueFollowUp(question.value);
      }
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    if (artifactViewer?.open) {
      return; // the dialog closes itself; don't also stop the run
    }
    if (stage.dataset.sessionsOpen === "1") {
      event.preventDefault();
      setSessionsOpen(false);
      return;
    }
    if (stage.dataset.statsOpen === "1") {
      event.preventDefault();
      setStatsOpen(false);
      return;
    }
    if (stage.dataset.running === "1") {
      event.preventDefault();
      void stopRun();
    }
  });

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = question.value;
    if (stage.dataset.running === "1") {
      void sendPrompt(text, { steer: true });
      return;
    }
    void sendPrompt(text);
  });

  updateStats();
  updateLatestButton();
  setRunning(false);
  renderProject();
  void loadProjects();
  void loadActivity();
  void probeHealth();
  void loadSessions();
  void restoreSession().then(() => {
    question.focus();
  });
})();
