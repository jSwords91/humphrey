(() => {
  const SESSION_KEY = "humphrey.sessionId";
  const CONTEXT_BUDGET = 128_000;

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
  const modelChip = document.getElementById("model-chip");
  const modelChipLabel = document.getElementById("model-chip-label");
  const sessionList = document.getElementById("session-list");
  const sessionsCount = document.getElementById("sessions-count");
  const activityDock = document.getElementById("activity-dock");
  const activityTitle = document.getElementById("activity-title");
  const activityList = document.getElementById("activity-list");
  const queueEl = document.getElementById("queue");
  const toastEl = document.getElementById("toast");
  const srAnnounce = document.getElementById("sr-announce");
  const latestBtn = document.getElementById("latest-btn");
  const latestBtnDot = document.getElementById("latest-btn-dot");
  const scrim = document.getElementById("scrim");
  const statModel = document.getElementById("stat-model");
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
  /** @type {{ kind: "steer" | "follow-up"; text: string }[]} */
  let queue = [];
  let pinnedToBottom = true;
  let hasNewBelow = false;
  let pinFrame = 0;
  /** @type {AbortController | null} */
  let activeRun = null;
  let suppressAbortMessage = false;
  let toolSeq = 0;
  let toolsUsed = 0;
  let charsIn = 0;
  let charsOut = 0;
  /** @type {number} */
  let tokensIn = 0;
  /** @type {number} */
  let tokensOut = 0;
  /** @type {Map<string, any>} */
  let liveTools = new Map();
  /** @type {number | null} */
  let activityTimer = null;
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
    const toggle = document.getElementById("sessions-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }
    syncScrim();
  }

  function setStatsOpen(open) {
    stage.dataset.statsOpen = open ? "1" : "0";
    const toggle = document.getElementById("stats-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
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
    const labels = { connecting: "Connecting", ready: "Ready", working: "Working" };
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

  /** @param {string} deployment */
  function setModelIdentity(deployment) {
    const label = (deployment || "").trim() || "unset";
    if (modelChipLabel) {
      modelChipLabel.textContent = label;
    }
    if (modelChip) {
      modelChip.title = label;
      modelChip.setAttribute("aria-label", `Model ${label}`);
    }
    if (statModel) {
      statModel.textContent = label;
      statModel.title = label;
    }
  }

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

  const MERMAID_THEME = {
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    themeVariables: {
      fontFamily: 'Geist, system-ui, -apple-system, "Segoe UI", sans-serif',
      primaryColor: "#ffffff",
      primaryTextColor: "#1a1a1a",
      primaryBorderColor: "rgba(0,0,0,0.14)",
      lineColor: "#737373",
      secondaryColor: "#f3f3f5",
      tertiaryColor: "#f7f7f8",
      clusterBkg: "#f7f7f8",
      clusterBorder: "#e7295d",
    },
  };

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
      mermaid.initialize(MERMAID_THEME);
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

  function updateStats() {
    const turns = messages.filter((m) => m.role === "user").length;
    const approxIn = tokensIn || Math.round(charsIn / 4);
    const approxOut = tokensOut || Math.round(charsOut / 4);
    const approxTokens = approxIn + approxOut;
    const pct = Math.min(100, Math.round((approxTokens / CONTEXT_BUDGET) * 100));

    if (statTurns) {
      statTurns.textContent = String(turns);
    }
    if (statTools) {
      statTools.textContent = String(toolsUsed);
    }
    if (statIo) {
      statIo.textContent = `${compactNum(approxIn)} / ${compactNum(approxOut)}`;
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

    // Live-turn-only preview (see openSession's reload path): the actual
    // image bytes aren't round-tripped back from the server on a session
    // reload, so this only ever renders for the message just sent in this
    // tab, from local blob/data URLs - never persisted or refetched.
    if (kind === "user" && opts.images && opts.images.length > 0) {
      const gallery = document.createElement("div");
      gallery.className = "msg-user-images";
      for (const url of opts.images) {
        const img = document.createElement("img");
        img.src = url;
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
    appendMessage("error", text);
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

  /**
   * @param {string} tool
   * @param {string} raw
   */
  function summarizeToolResult(tool, raw) {
    try {
      const data = JSON.parse(raw);
      if (data && typeof data === "object" && data.error) {
        return String(data.error).slice(0, 120);
      }
      if (tool === "list_docs" && Array.isArray(data.documents)) {
        return `${data.documents.length} document${data.documents.length === 1 ? "" : "s"}`;
      }
      if (tool === "search_docs") {
        const n = Number(data.total_matches ?? data.hits?.length ?? 0);
        return `${n} match${n === 1 ? "" : "es"}`;
      }
      if (tool === "read_doc" && data.path) {
        return `${data.path} · lines ${data.start_line}–${data.end_line}`;
      }
      if (tool === "get_weather" && data.place) {
        const temp = data.current?.temperature_2m;
        return temp == null ? String(data.place) : `${data.place} · ${temp}°C`;
      }
      if (tool === "calculate" && "result" in data) {
        return `= ${data.result}`;
      }
      if (tool === "status") {
        const model = data.model || {};
        const dep = model.openai?.model || model.azure?.deployment || "unset";
        return data.mock_mode ? `mock · ${dep}` : String(dep);
      }
      if (tool === "capabilities" && Array.isArray(data.tools)) {
        return `${data.tools.length} tools`;
      }
    } catch {
      /* fall through */
    }
    const flat = raw.replace(/\s+/g, " ").trim();
    return flat.length > 72 ? `${flat.slice(0, 69)}…` : flat || "done";
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

  function refreshActivityDock() {
    const running = [...liveTools.values()].filter((t) => t.el.dataset.state === "running");
    if (!running.length) {
      activityDock.hidden = true;
      activityList.replaceChildren();
      if (activityTimer) {
        window.clearInterval(activityTimer);
        activityTimer = null;
      }
      if (lastToolSummary.endsWith(" · running")) {
        setToolStat(lastToolSummary.replace(/ · running$/, " · done"));
      }
      return;
    }

    activityDock.hidden = false;
    activityTitle.textContent = `${running.length} tool${running.length === 1 ? "" : "s"} running`;
    activityList.replaceChildren();

    const primary = running[running.length - 1];
    setToolStat(`${primary.display} · running`);

    for (const card of running) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "activity-row";
      const name = document.createElement("span");
      name.className = "activity-name";
      name.textContent = card.display;
      const preview = document.createElement("span");
      preview.className = "activity-preview";
      preview.textContent = card.preview || "…";
      const elapsed = document.createElement("span");
      elapsed.className = "activity-elapsed";
      elapsed.textContent = `${Math.max(1, Math.round(performance.now() - card.started))} ms`;
      btn.append(name, preview, elapsed);
      btn.addEventListener("click", () => focusToolCard(card.el));
      li.append(btn);
      activityList.append(li);
    }

    if (!activityTimer) {
      activityTimer = window.setInterval(() => refreshActivityDock(), 250);
    }
  }

  /** @param {HTMLElement} el */
  function focusToolCard(el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("is-flash");
    window.setTimeout(() => el.classList.remove("is-flash"), 900);
    window.setTimeout(() => syncPinFromScroll(), 400);
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
   * Repeat calls to the same tool within one turn (e.g. get_weather
   * for several places) collapse into one cluster
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
    const signature = toolSignature(display, args);

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
    refreshActivityDock();
    return card;
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
   * dot goes on visibly pulsing pink long after the row reads "done"/"error".
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
   * data-state="running" - left alone, its dot pulses pink forever even
   * though the run itself is long over. Called once per run, from the
   * sendPrompt finally block, to settle anything still marked running.
   * @param {{ toolEls: Map<string, any>, toolClusters: Map<string, any> }} state
   */
  function finalizeStuckTools(state) {
    for (const card of state.toolEls.values()) {
      if (card.el.dataset.state === "running") {
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
    if (activeRun) {
      suppressAbortMessage = true;
      activeRun.abort();
      activeRun = null;
    }
    messages = [];
    sessionId = null;
    localStorage.removeItem(SESSION_KEY);
    queue = [];
    renderQueue();
    liveTools.clear();
    refreshActivityDock();
    stopThoughtTimer();
    pinnedToBottom = true;
    hasNewBelow = false;
    toolsUsed = 0;
    charsIn = 0;
    charsOut = 0;
    tokensIn = 0;
    tokensOut = 0;
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
    try {
      const res = await fetch("/api/sessions");
      if (!res.ok) {
        return;
      }
      /** @type {{ id: string, title?: string, message_count?: number, updated_at?: string }[]} */
      const rows = await res.json();
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
        empty.textContent = "No recent chats";
        sessionList.append(empty);
        return;
      }
      for (const row of rows) {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "session-item";
        if (row.id === sessionId) {
          btn.classList.add("is-active");
        }
        const title = document.createElement("span");
        title.className = "session-title";
        title.textContent = row.title || "Untitled chat";
        const meta = document.createElement("span");
        meta.className = "session-meta";
        meta.textContent = relativeTime(row.updated_at) || "Saved";
        btn.append(title, meta);
        btn.addEventListener("click", () => {
          void openSession(row.id);
        });
        li.append(btn);
        sessionList.append(li);
      }
    } catch {
      /* ignore */
    }
  }

  /**
   * @param {string} id
   * @param {{ silent?: boolean }} [opts]
   */
  async function openSession(id, opts = {}) {
    if (activeRun) {
      activeRun.abort();
      activeRun = null;
    }
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
      const rows = (data.messages || []).filter((m) => m && m.content);
      // Only user/assistant turns are ever resent to /api/agent (ChatMessage
      // on the wire has no "context_summary" role) - a context-summary row
      // is rendered below but kept out of this array.
      messages = rows.filter((m) => m.role === "user" || m.role === "assistant");
      sessionId = id;
      localStorage.setItem(SESSION_KEY, id);
      clearTranscript();
      liveTools.clear();
      refreshActivityDock();
      toolsUsed = 0;
      charsIn = 0;
      charsOut = 0;
      // Real usage isn't attached to individual messages in this response
      // (only the last assistant turn's cumulative usage is), so seed the
      // real counters from that and let the per-message loop below only
      // fill in the char-based fallback for sessions with no usage yet
      // (e.g. mock mode, or a session with no assistant reply).
      tokensIn = 0;
      tokensOut = 0;
      if (data.last_usage && typeof data.last_usage === "object") {
        const u = data.last_usage;
        if (typeof u.input === "number") {
          tokensIn = u.input;
        }
        if (typeof u.output === "number") {
          tokensOut = u.output;
        }
      }
      setToolStat("—");
      for (const msg of rows) {
        if (msg.role === "context_summary") {
          appendMessage("context-summary", msg.content, { animate: false });
          continue;
        }
        if (msg.role === "user") {
          charsIn += msg.content.length;
        } else {
          charsOut += msg.content.length;
        }
        appendMessage(msg.role, msg.content, { animate: false });
      }
      setActive(messages.length > 0);
      setRunning(false);
      updateStats();
      pinnedToBottom = true;
      hasNewBelow = false;
      updateLatestButton();
      pinToBottom();
      void loadSessions();
      question.focus();
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
      const providerLabels = {
        mock: "mock",
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
      const deployment =
        typeof data.deployment === "string" && data.deployment.trim()
          ? data.deployment.trim()
          : data.mock_mode
            ? "mock"
            : "unset";
      setModelIdentity(deployment);
      if (statProvider) {
        statProvider.textContent = provider;
      }
      setStatus(stage.dataset.running === "1" ? "working" : "ready");
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

    if (type === "agent_start") {
      if (typeof event.session_id === "string" && event.session_id) {
        sessionId = event.session_id;
        localStorage.setItem(SESSION_KEY, sessionId);
      }
      setRunning(true);
      ensureThought(state);
      return;
    }

    if (type === "turn_start") {
      setRunning(true);
      state.turn = typeof event.turn === "number" ? event.turn : null;
      state.toolGroupEl = null;
      state.toolGroupCount = 0;
      state.toolClusters = new Map();
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
      if (typeof event.deployment === "string" && event.deployment.trim()) {
        setModelIdentity(event.deployment.trim());
      }
      if (event.usage && typeof event.usage === "object") {
        // Each turn's usage.input is already a cumulative count (the whole
        // transcript is resent every call), so the latest turn's numbers ARE
        // the current context-window occupancy - replace, don't accumulate,
        // or multi-turn conversations wildly over-report their own size.
        const u = event.usage;
        if (typeof u.input === "number") {
          tokensIn = u.input;
        }
        if (typeof u.output === "number") {
          tokensOut = u.output;
        }
      }
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
        updateThoughtBody(state, `Using ${label}…`);
      }
      const card = appendToolRow(tool, callId, event.args ?? {}, label, state);
      state.toolEls.set(callId, card);
      setRunning(true);
      return;
    }

    if (type === "tool_end") {
      const callId = String(event.call_id ?? "");
      const tool = String(event.tool ?? "tool");
      const label = typeof event.label === "string" ? event.label : tool;
      const rawResult = String(event.result ?? "");
      const result = prettyResult(rawResult);
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
      const blurb = summarizeToolResult(tool, rawResult);
      const isError = event.is_error === true;
      card.meta.textContent = `${ms} ms`;
      card.el.dataset.state = isError ? "error" : "done";
      stopDotPulse(card.el);
      card.preview = blurb;
      card.toggle.setAttribute(
        "aria-label",
        `${card.display}, ${blurb}, ${ms} milliseconds. Activate to show full output.`,
      );
      card.output.textContent = result;

      const userToggled = card.el.dataset.userToggled === "1";
      if (!userToggled) {
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
      if (callId) {
        liveTools.delete(callId);
      }
      setToolStat(`${card.display} · ${isError ? "error" : "done"} · ${ms} ms`);
      refreshActivityDock();
      pinToBottom();
      return;
    }

    if (type === "error") {
      settleThought(state);
      showError(String(event.detail ?? "Unknown agent error"));
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

    if (activeRun && sessionId && !opts.fromQueue) {
      const kind = opts.steer ? "steer" : "follow-up";
      if (await postQueue(kind, text)) {
        return;
      }
    }

    if (activeRun && !opts.steer && !opts.fromQueue) {
      enqueueFollowUp(text);
      return;
    }

    if (activeRun) {
      if (opts.steer) {
        suppressAbortMessage = true;
      }
      activeRun.abort();
      activeRun = null;
    }

    const controller = new AbortController();
    activeRun = controller;

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
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          messages,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });
      // Images are only ever read from the *last* message server-side
      // (app/main.py's /api/agent) - strip them here so the next turn's
      // full-transcript resend doesn't re-upload the same bytes forever.
      delete pushedMessage.images;

      const headerSession = res.headers.get("X-Session-Id");
      if (headerSession) {
        sessionId = headerSession;
        localStorage.setItem(SESSION_KEY, sessionId);
      }

      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const data = await res.json();
          detail =
            typeof data.detail === "string"
              ? data.detail
              : JSON.stringify(data.detail ?? data, null, 2);
        } catch {
          /* keep status */
        }
        messages.pop();
        transcript.querySelector(".msg-user:last-of-type")?.remove();
        if (messages.length === 0) {
          setActive(false);
        }
        showError(detail);
        return;
      }

      if (!res.body) {
        showError("No response body from /api/agent");
        return;
      }

      await readSse(res.body, (ev) => applyEvent(ev, state), controller.signal);
    } catch (err) {
      if (err && typeof err === "object" && "name" in err && err.name === "AbortError") {
        if (suppressAbortMessage) {
          suppressAbortMessage = false;
        } else {
          appendMessage("status", "Stopped.");
          announce("Run stopped.");
        }
        return;
      }
      showError(String(err));
    } finally {
      if (activeRun === controller) {
        activeRun = null;
        finalizeStuckTools(state);
        liveTools.clear();
        refreshActivityDock();
        settleThought(state);
        setRunning(false);
        void loadSessions();
        question.focus();
        void drainQueue();
      }
    }
  }

  function stopRun() {
    if (activeRun) {
      suppressAbortMessage = false;
      activeRun.abort();
    }
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
  stopBtn?.addEventListener("click", () => stopRun());
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
      stopRun();
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
  void probeHealth();
  void loadSessions();
  void restoreSession().then(() => {
    question.focus();
  });
})();
