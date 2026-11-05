/* Floating chat widget. Add <script src="/chat.js" defer></script> before </body>. */
(() => {
  const css = `
#wq-chat{position:fixed;right:20px;bottom:20px;z-index:60;font:14px/1.5 "Instrument Sans",system-ui,sans-serif;color:#1d4a44}
#wq-chat *{box-sizing:border-box}
.wc-fab{width:56px;height:56px;border-radius:50%;border:0;background:#1d4a44;color:#fff;cursor:pointer;display:grid;place-items:center;box-shadow:0 8px 24px rgba(29,74,68,.3);transition:transform .2s}
.wc-fab:hover{transform:scale(1.06)}
.wc-fab:focus-visible,#wq-chat button:focus-visible,#wq-chat input:focus-visible{outline:2px solid #c9769b;outline-offset:2px}
.wc-panel{position:absolute;right:0;bottom:70px;width:min(370px,calc(100vw - 32px));height:min(520px,calc(100vh - 110px));display:flex;flex-direction:column;background:rgba(255,255,255,.96);border:1px solid rgba(29,74,68,.14);border-radius:18px;box-shadow:0 18px 50px rgba(29,74,68,.22);overflow:hidden}
.wc-panel[hidden]{display:none}
.wc-panel header{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;background:#e6f2ee;border-bottom:1px solid rgba(29,74,68,.12)}
.wc-x{border:0;background:none;font-size:22px;line-height:1;color:#4d726b;cursor:pointer;padding:0 4px}
.wc-log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
.wc-m{max-width:86%;padding:9px 13px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
.wc-m.bot{align-self:flex-start;background:#eef5f2}
.wc-m.me{align-self:flex-end;background:#1d4a44;color:#fff}
.wc-chips{display:flex;flex-wrap:wrap;gap:6px;padding:0 14px 10px}
.wc-chips button{border:1px solid rgba(29,74,68,.2);background:#fff;color:#1d4a44;border-radius:999px;padding:5px 11px;font:inherit;font-size:12.5px;cursor:pointer}
.wc-chips button:hover{border-color:#c9769b}
.wc-form{display:flex;gap:8px;padding:10px 12px;border-top:1px solid rgba(29,74,68,.12)}
.wc-form input{flex:1;min-width:0;border:1px solid rgba(29,74,68,.2);border-radius:999px;padding:9px 14px;font:inherit;color:inherit}
.wc-form button{border:0;border-radius:999px;background:#1d4a44;color:#fff;padding:0 16px;font:inherit;font-weight:600;cursor:pointer}
.wc-form button:disabled{opacity:.5;cursor:default}
@media (prefers-reduced-motion:reduce){.wc-fab{transition:none}}`;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);

  const root = document.createElement('div'); root.id = 'wq-chat';
  root.innerHTML = `
<button class="wc-fab" aria-label="Open chat" aria-expanded="false"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.5 7.2L4 20.5l1.4-4.6A8 8 0 1 1 21 12Z"/></svg></button>
<section class="wc-panel" hidden role="dialog" aria-label="Water quality assistant">
<header><b>Water quality assistant</b><button class="wc-x" aria-label="Close chat">&times;</button></header>
<div class="wc-log" aria-live="polite"></div><div class="wc-chips"></div>
<form class="wc-form"><input maxlength="500" placeholder="Ask about WQI, models, rivers..." aria-label="Message" autocomplete="off"><button type="submit">Send</button></form></section>`;
  document.body.appendChild(root);

  const $ = s => root.querySelector(s);
  const fab = $('.wc-fab'), panel = $('.wc-panel'), log = $('.wc-log'), input = $('.wc-form input'), send = $('.wc-form button'), chips = $('.wc-chips');
  const hist = []; let busy = false;

  const add = (cls, text) => { const d = document.createElement('div'); d.className = 'wc-m ' + cls; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; };
  const toggle = open => { panel.hidden = !open; fab.setAttribute('aria-expanded', open); if (open) input.focus(); else fab.focus(); };

  add('bot', 'Hi! Ask me about the Water Quality Index, the forecasting models, or how to read this dashboard.');
  ['What does WQI mean?', 'Why is a river Slightly Polluted?', 'Which model is best?'].forEach(q => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = q; b.onclick = () => ask(q); chips.appendChild(b);
  });

  async function ask(text) {
    text = text.trim(); if (!text || busy) return;
    busy = true; send.disabled = true; chips.hidden = true; input.value = '';
    add('me', text); hist.push({ role: 'user', content: text });
    const node = add('bot', 'Thinking...');
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: hist }) });
      const d = await r.json().catch(() => ({}));
      if (!r.ok || !d.reply) throw new Error('HTTP ' + r.status + ': ' + (d.error || 'no JSON reply (is /api/chat deployed?)'));
      node.textContent = d.reply; hist.push({ role: 'assistant', content: d.reply });
    } catch (e) {
      console.error('[chat]', e);
      node.textContent = 'Sorry, I could not reply just now. Please try again in a moment.'; hist.pop();
    }
    busy = false; send.disabled = false; log.scrollTop = log.scrollHeight; input.focus();
  }

  fab.onclick = () => toggle(panel.hidden);
  $('.wc-x').onclick = () => toggle(false);
  $('.wc-form').onsubmit = e => { e.preventDefault(); ask(input.value); };
  root.addEventListener('keydown', e => { if (e.key === 'Escape' && !panel.hidden) toggle(false); });
})();