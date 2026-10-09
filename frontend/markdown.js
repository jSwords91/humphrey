/**
 * XSS-safe Markdown subset for assistant replies.
 * Supports: paragraphs, headings, emphasis, strikethrough, inline/fenced code,
 * nested lists, GFM tables, links, blockquotes, horizontal rules, mermaid fences.
 */
(() => {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
  const KW =
    /\b(const|let|var|function|return|if|else|for|while|class|import|from|export|async|await|def|True|False|None|and|or|not|in|as|with|try|except|raise|pass|yield|true|false|null|None)\b/g;
  const NUM = /\b(\d+(?:\.\d+)?)\b/g;

  /** @param {string} text */
  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (ch) => ESC[ch] || ch);
  }

  /**
   * @param {string} escaped
   * @returns {string}
   */
  function highlightGeneric(escaped) {
    const re = /(&quot;[\s\S]*?&quot;)/g;
    let out = "";
    let last = 0;
    let m;
    while ((m = re.exec(escaped))) {
      out += escaped
        .slice(last, m.index)
        .replace(KW, '<span class="tok-kw">$1</span>')
        .replace(NUM, '<span class="tok-num">$1</span>');
      out += `<span class="tok-str">${m[1]}</span>`;
      last = m.index + m[1].length;
    }
    out += escaped
      .slice(last)
      .replace(KW, '<span class="tok-kw">$1</span>')
      .replace(NUM, '<span class="tok-num">$1</span>');
    return out;
  }

  /**
   * @param {string} escaped
   * @returns {string}
   */
  function highlightJson(escaped) {
    const re =
      /&quot;(?:[^&]|&(?!quot;))*&quot;|\b-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b(?:true|false|null)\b/g;
    let out = "";
    let last = 0;
    let m;
    while ((m = re.exec(escaped))) {
      out += escaped.slice(last, m.index);
      const tok = m[0];
      if (tok.startsWith("&quot;")) {
        const after = escaped.slice(m.index + tok.length);
        out += /^\s*:/.test(after)
          ? `<span class="tok-key">${tok}</span>`
          : `<span class="tok-str">${tok}</span>`;
      } else if (/^-?\d/.test(tok)) {
        out += `<span class="tok-num">${tok}</span>`;
      } else {
        out += `<span class="tok-kw">${tok}</span>`;
      }
      last = m.index + tok.length;
    }
    return out + escaped.slice(last);
  }

  /**
   * @param {string} code
   * @param {string} lang
   */
  function highlightCode(code, lang) {
    const escaped = escapeHtml(code);
    const kind = (lang || "").toLowerCase();
    if (kind === "json") {
      return highlightJson(escaped);
    }
    return highlightGeneric(escaped);
  }

  /** @param {string} text */
  function inline(text) {
    let s = escapeHtml(text);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_\w])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
    s = s.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" rel="noopener noreferrer" target="_blank">$1</a>',
    );
    return s;
  }

  /** @param {string} line */
  function isTableSep(line) {
    return /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/.test(line);
  }

  /** @param {string} line */
  function splitCells(line) {
    let t = line.trim();
    if (t.startsWith("|")) {
      t = t.slice(1);
    }
    if (t.endsWith("|")) {
      t = t.slice(0, -1);
    }
    return t.split("|").map((cell) => cell.trim());
  }

  /**
   * @param {string} cell
   * @returns {"c" | "r" | ""}
   */
  function alignFromSep(cell) {
    const t = cell.trim();
    const left = t.startsWith(":");
    const right = t.endsWith(":");
    if (left && right) {
      return "c";
    }
    if (right) {
      return "r";
    }
    return "";
  }

  /**
   * @param {string[]} header
   * @param {string[]} aligns
   * @param {string[][]} rows
   */
  function renderTable(header, aligns, rows) {
    const th = header
      .map((cell, i) => {
        const al = aligns[i] ? ` class="md-al-${aligns[i]}"` : "";
        return `<th${al}>${inline(cell)}</th>`;
      })
      .join("");
    const body = rows
      .map((row) => {
        const tds = header
          .map((_, i) => {
            const al = aligns[i] ? ` class="md-al-${aligns[i]}"` : "";
            return `<td${al}>${inline(row[i] || "")}</td>`;
          })
          .join("");
        return `<tr>${tds}</tr>`;
      })
      .join("");
    return `<div class="md-table"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  /**
   * @param {string} source
   * @returns {string}
   */
  function renderMarkdown(source) {
    const lines = String(source).replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let i = 0;
    /** @type {{ tag: "ul" | "ol", indent: number, openLi: boolean }[]} */
    const listStack = [];
    let para = [];

    function closeLists(toIndent = -1) {
      while (listStack.length && listStack[listStack.length - 1].indent > toIndent) {
        const item = listStack.pop();
        if (item && item.openLi) {
          html.push("</li>");
        }
        if (item) {
          html.push(`</${item.tag}>`);
        }
      }
    }

    function closeAllLists() {
      closeLists(-1);
    }

    function flushPara() {
      if (!para.length) {
        return;
      }
      html.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    }

    /**
     * @param {"ul" | "ol"} tag
     * @param {number} indent
     * @param {string} content
     */
    function pushListItem(tag, indent, content) {
      if (!listStack.length || indent > listStack[listStack.length - 1].indent) {
        html.push(`<${tag}>`);
        listStack.push({ tag, indent, openLi: false });
      } else {
        closeLists(indent);
        const top = listStack[listStack.length - 1];
        if (!top || top.indent < indent) {
          html.push(`<${tag}>`);
          listStack.push({ tag, indent, openLi: false });
        } else if (top.tag !== tag) {
          if (top.openLi) {
            html.push("</li>");
          }
          html.push(`</${top.tag}>`);
          listStack.pop();
          html.push(`<${tag}>`);
          listStack.push({ tag, indent, openLi: false });
        } else if (top.openLi) {
          html.push("</li>");
          top.openLi = false;
        }
      }
      const current = listStack[listStack.length - 1];
      html.push(`<li>${inline(content)}`);
      current.openLi = true;
    }

    while (i < lines.length) {
      const line = lines[i];

      if (line.startsWith("```")) {
        flushPara();
        closeAllLists();
        const lang = line.slice(3).trim() || "code";
        i += 1;
        const code = [];
        let closed = false;
        while (i < lines.length) {
          if (lines[i].startsWith("```")) {
            closed = true;
            i += 1;
            break;
          }
          code.push(lines[i]);
          i += 1;
        }
        const source = code.join("\n");
        if (closed && isMermaidFence(lang, source)) {
          html.push(
            `<div class="md-mermaid"><pre class="md-mermaid-src">${escapeHtml(source)}</pre></div>`,
          );
          continue;
        }
        const body = highlightCode(source, lang);
        const pending = closed ? "" : ' data-pending="1"';
        html.push(
          `<div class="code-block"${pending}><div class="code-block-head"><span>${escapeHtml(lang)}</span><button type="button" class="code-copy">Copy</button></div><pre><code>${body}</code></pre></div>`,
        );
        continue;
      }

      if (/^\s*$/.test(line)) {
        flushPara();
        i += 1;
        continue;
      }

      if (/^ {0,3}([-*_])\1{2,}\s*$/.test(line)) {
        flushPara();
        closeAllLists();
        html.push("<hr />");
        i += 1;
        continue;
      }

      const heading = /^(#{1,3})\s+(.+)$/.exec(line);
      if (heading) {
        flushPara();
        closeAllLists();
        const level = heading[1].length;
        html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        i += 1;
        continue;
      }

      if (
        line.includes("|") &&
        i + 1 < lines.length &&
        isTableSep(lines[i + 1])
      ) {
        flushPara();
        closeAllLists();
        const header = splitCells(line);
        const aligns = splitCells(lines[i + 1]).map(alignFromSep);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && !/^\s*$/.test(lines[i])) {
          if (lines[i].startsWith("```") || /^(#{1,3})\s+/.test(lines[i])) {
            break;
          }
          rows.push(splitCells(lines[i]));
          i += 1;
        }
        html.push(renderTable(header, aligns, rows));
        continue;
      }

      const quote = /^>\s?(.*)$/.exec(line);
      if (quote) {
        flushPara();
        closeAllLists();
        const chunks = [quote[1]];
        i += 1;
        while (i < lines.length) {
          const next = /^>\s?(.*)$/.exec(lines[i]);
          if (!next) {
            break;
          }
          chunks.push(next[1]);
          i += 1;
        }
        html.push(`<blockquote><p>${inline(chunks.join(" "))}</p></blockquote>`);
        continue;
      }

      const list = /^(\s*)([-*]|\d+\.)\s+(.+)$/.exec(line);
      if (list) {
        flushPara();
        const indent = list[1].length;
        const tag = /[-*]/.test(list[2]) ? "ul" : "ol";
        pushListItem(tag, indent, list[3]);
        i += 1;
        continue;
      }

      if (listStack.length) {
        const indent = (/^(\s*)/.exec(line) || ["", ""])[1].length;
        const top = listStack[listStack.length - 1];
        if (indent > top.indent) {
          html.push(` ${inline(line.trim())}`);
          i += 1;
          continue;
        }
      }

      closeAllLists();
      para.push(line.trim());
      i += 1;
    }

    flushPara();
    closeAllLists();
    return html.join("") || `<p>${inline(source)}</p>`;
  }

  /**
   * @param {string} lang
   * @param {string} source
   */
  function isMermaidFence(lang, source) {
    const head = lang.toLowerCase().split(/[\s,]+/)[0] || "";
    if (head === "mermaid") {
      return true;
    }
    if (head && head !== "code") {
      return false;
    }
    return /^(?:flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|gantt|pie|mindmap|timeline|gitGraph|C4Context|quadrantChart|sankey-beta|xychart-beta|block-beta|packet-beta|kanban|architecture-beta)\b/.test(
      source.trim(),
    );
  }

  window.HumphreyMarkdown = { render: renderMarkdown, escapeHtml, isMermaidFence };
})();
