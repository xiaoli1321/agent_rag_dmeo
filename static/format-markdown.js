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
      
      // Unwrap backtick-wrapped markdown links that contain media URLs
      html = html.replace(/`\[([^\]]*)\]\(([^)]+)\)`/g, (match, text, url) => {
        if (/\.(?:png|jpe?g|gif|webp|svg|bmp|mp4)(?:[?#/]|$)/i.test(url)) {
          return `[${text}](${url})`;
        }
        return match;
      });
      
      // Inline code
      html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
      
      // Bold
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      
      // Markdown images ![alt](url) — common image formats
      html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {
        if (/\.(png|jpe?g|gif|webp|svg|bmp)(?:[?#/]|$)/i.test(url)) {
          return `<img src="${escapeAttr(url)}" alt="${escapeAttr(alt)}" loading="lazy" class="chat-image">`;
        }
        return `<a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${escapeAttr(alt || url)}</a>`;
      });
      
      // Markdown links [text](url) or [](url) — MP4 URLs → embedded video
      html = html.replace(/\[([^\]]*)\]\(([^)]+)\)/g, (match, text, url) => {
        if (/\.mp4(?:[?#/]|$)/i.test(url)) {
          return `<video controls playsinline preload="metadata" src="${escapeAttr(url)}"></video>`;
        }
        return `<a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${escapeAttr(text || url)}</a>`;
      });
      
      // Protect generated HTML tags from bare URL regex
      const protectedHTML = {};
      let protectIdx = 0;
      html = html.replace(/(<(?:a|video|img)\s[^>]*>)/g, (match) => {
        const key = `\x00PR${protectIdx++}\x00`;
        protectedHTML[key] = match;
        return key;
      });
      
      // Bare URLs → clickable links, embedded video, or images
      html = html.replace(/(https?:\/\/[^\s<"]+)/g, (match, url) => {
        if (/\.(png|jpe?g|gif|webp|svg|bmp)(?:[?#/]|$)/i.test(url)) {
          return `<img src="${escapeAttr(url)}" alt="" loading="lazy" class="chat-image">`;
        }
        if (/\.mp4(?:[?#/]|$)/i.test(url)) {
          return `<video controls playsinline preload="metadata" src="${escapeAttr(url)}"></video>`;
        }
        return `<a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${url}</a>`;
      });
      
      // Restore protected HTML tags
      html = html.replace(/\x00PR\d+\x00/g, (key) => protectedHTML[key] || key);
      
      // Lists
      const lines = html.split('\n');
      let inList = false;
      const processedLines = [];
      
      for (let line of lines) {
        const listMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
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
      
      html = processedLines.join('\n');

      // Collapse consecutive blank lines to single line breaks
      html = html.replace(/\n+/g, '\n');

      // Strip newlines around block elements to avoid double spacing with pre-wrap
      html = html.replace(/\n+(?=\s*<(?:ul|ol|pre|blockquote|h[1-6])[^>]*>)/g, '');
      html = html.replace(/(<\/(?:ul|ol|pre|blockquote|h[1-6])>)\s*\n+/g, '$1');

      return html;
    }

    function escapeAttr(s) {
      return s.replace(/"/g, "&quot;");
    }
