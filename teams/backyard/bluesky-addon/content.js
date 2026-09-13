/* ============================================================
   Trust Score for Bluesky — content script.
   - finds every profile link on bsky.app (feed, thread, followers lists, ...)
   - accounts near the viewport are requested with priority 0, the ones further down with priority 1
     ("upcoming users"); the server crawls them breadth-first and scores the graph
   - a badge (same look as the Reddit mock-up) is injected after the account name; hover/click opens the
     explanation popover with seed buttons (mark trusted / untrusted) that update the propagation
   Works without the extension runtime too (dev mode: direct fetch to the server), so it can be injected
   into a plain tab for testing.
   ============================================================ */
(function () {
  if (window.__tsBlueskyLoaded) return;
  window.__tsBlueskyLoaded = true;

  const SHIELD = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5 4.5 5.5v6c0 5 3.3 8.6 7.5 10 4.2-1.4 7.5-5 7.5-10v-6z"/><path d="m9 12 2 2 4-4"/></svg>';
  const BOLT = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>';
  const hasRuntime = typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.id;
  let serverBase = (window.__tsServer || 'http://127.0.0.1:8010').replace(/\/$/, '');
  let enabled = true;

  // ---------- API ----------
  async function api(path, method, body) {
    if (hasRuntime) {
      const res = await new Promise((resolve) => chrome.runtime.sendMessage({ type: 'api', path, method, body }, resolve));
      if (!res) throw new Error('no response from background');
      if (!res.ok) throw new Error(res.error || ('HTTP ' + res.status));
      return res.data;
    }
    const r = await fetch(serverBase + path, { method: method || 'GET', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }

  const dashUrl = (q) => serverBase + '/addon/dashboard.html' + (q || '');
  function visibleIdents(limit) {
    const out = [];
    badges.forEach((set, id) => {
      for (const el of set) { const r = el.getBoundingClientRect(); if (r.bottom > 0 && r.top < window.innerHeight) { out.push(id); break; } }
    });
    if (!out.length) badges.forEach((_, id) => out.push(id));
    return out.slice(0, limit || 20);
  }

  // ---------- state ----------
  const accounts = new Map();        // ident (lowercase handle or did) -> payload from the server
  const pending = new Map();         // ident -> priority (waiting to be sent)
  const badges = new Map();          // ident -> Set(badge elements)
  let flushTimer = null, pollTimer = null, online = null;
  const dark = () => document.documentElement.getAttribute('data-theme') === 'dark' || matchMedia('(prefers-color-scheme: dark)').matches
    || getComputedStyle(document.body).backgroundColor === 'rgb(0, 0, 0)' || getComputedStyle(document.body).backgroundColor === 'rgb(22, 30, 39)';

  function identOf(a) {
    const m = a.getAttribute('href') && a.getAttribute('href').match(/^\/profile\/([^/?#]+)/);
    if (!m) return null;
    let id = decodeURIComponent(m[1]);
    if (!id.startsWith('did:')) id = id.toLowerCase();
    return id;
  }

  // ---------- badge rendering ----------
  function cls(a) {
    if (!a) return 'ts-pending';
    if (a.sparse) return 'ts-grey';
    if (a.score == null) return a.state === 'error' ? 'ts-grey' : 'ts-pending';
    if (a.score >= 0.70) return 'ts-green';
    if (a.score >= 0.40) return 'ts-amber';
    return 'ts-red';
  }
  function verdict(a) {
    if (a && a.sparse) return 'Not enough graph';
    if (!a || a.score == null) return a && a.state === 'error' ? 'could not crawl' : 'crawling…';
    if (a.score >= 0.70) return 'Likely genuine';
    if (a.score >= 0.40) return 'Uncertain';
    return 'Bot-like';
  }
  function badgeText(a) {
    if (a && a.sparse) return '?';
    if (!a || a.score == null) return a && a.state === 'error' ? 'n/a' : '…';
    return Math.round(a.score * 100);
  }
  function paint(el, a) {
    el.className = 'ts-badge ' + cls(a);
    el.innerHTML = SHIELD + '<span class="ts-num">' + badgeText(a) + '</span>';
    el.title = 'Trust Score' + (a && a.handle ? ' for @' + a.handle : '') + (a && a.sparse ? ' — not enough graph yet (crawl queued)' : ' — click for the explanation');
  }
  function repaint(ident) {
    const set = badges.get(ident); if (!set) return;
    const a = accounts.get(ident);
    set.forEach((el) => { if (!el.isConnected) set.delete(el); else paint(el, a); });
  }

  function makeBadge(ident) {
    const el = document.createElement('span');
    el.dataset.ts = ident;
    paint(el, accounts.get(ident));
    el.setAttribute('role', 'button'); el.setAttribute('tabindex', '0');
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    el.addEventListener('click', (e) => { stop(e); toggle(el, ident); });
    el.addEventListener('mousedown', stop); el.addEventListener('mouseup', stop);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { stop(e); toggle(el, ident); } });
    el.addEventListener('mouseenter', () => { clearTimeout(hoverTimer); hoverTimer = setTimeout(() => open(el, ident), 300); });
    el.addEventListener('mouseleave', () => { clearTimeout(hoverTimer); hoverTimer = setTimeout(() => { if (!pop || !pop.matches(':hover')) close(); }, 250); });
    if (!badges.has(ident)) badges.set(ident, new Set());
    badges.get(ident).add(el);
    return el;
  }

  // ---------- page scanning ----------
  // One badge per (post/list item, account). The anchor that carries the visible name is the one whose text is
  // not empty and that is not purely an avatar; bsky.app renders "Display Name @handle" as text inside the link.
  function containerOf(a) {
    return a.closest('[data-testid^="feedItem"], [data-testid^="postThreadItem"], [data-testid^="notification"], [data-testid^="user-"], [data-testid="profileHeaderDisplayName"], [role="article"], [role="listitem"]')
      || a.parentElement;
  }
  function scan() {
    if (!enabled) return;
    const near = window.innerHeight + 1500;
    document.querySelectorAll('a[href^="/profile/"]').forEach((a) => {
      if (a.dataset.tsSeen) return;
      const ident = identOf(a); if (!ident) return;
      const href = a.getAttribute('href');
      if (/\/profile\/[^/]+\/(post|feed|lists|follow|starter)/.test(href)) return;   // links to posts/lists, not to the account
      const text = (a.textContent || '').replace(/\u00a0/g, ' ').trim();
      if (!text) return;                                                           // avatar-only link
      a.dataset.tsSeen = '1';
      const box = containerOf(a);
      const isHandle = text.startsWith('@');
      const existing = box && box.querySelector('[data-ts="' + CSS.escape(ident) + '"]');
      if (existing) {
        if (isHandle) a.insertAdjacentElement('afterend', existing);   // prefer "Name @handle [badge]" over "Name [badge] @handle"
        return;
      }
      const badge = makeBadge(ident);
      // Inside a card-like anchor (profile cards, follower lists) put the badge right after the name element.
      const inner = a.childElementCount ? [...a.querySelectorAll('div[dir="auto"], span')].find((el) => el.childElementCount === 0 && el.textContent.trim()) : null;
      if (inner && inner.parentElement) inner.parentElement.insertBefore(badge, inner.nextSibling);
      else a.insertAdjacentElement('afterend', badge);
      const top = a.getBoundingClientRect().top;
      const prio = top >= -200 && top <= near ? 0 : 1;
      const known = accounts.get(ident);
      if (!known || known.score == null || known.sparse) queue(ident, prio);
    });
  }

  function queue(ident, prio) {
    const cur = pending.get(ident);
    if (cur == null || prio < cur) pending.set(ident, prio);
    clearTimeout(flushTimer); flushTimer = setTimeout(flush, 350);
  }
  async function flush() {
    if (!pending.size) return;
    const byPrio = [[], []];
    pending.forEach((p, id) => byPrio[p].push(id));
    pending.clear();
    for (let p = 0; p < 2; p++) {
      if (!byPrio[p].length) continue;
      try {
        const res = await api('/bsky/lookup', 'POST', { idents: byPrio[p].slice(0, 200), priority: p });
        setOnline(true);
        Object.entries(res.accounts || {}).forEach(([id, a]) => { accounts.set(id.toLowerCase(), a || { state: 'error' }); repaint(id.toLowerCase()); });
      } catch (e) { setOnline(false, e); }
    }
    schedulePoll();
  }
  function schedulePoll() {
    clearTimeout(pollTimer);
    const waiting = [...badges.keys()].filter((id) => { const a = accounts.get(id); return !a || (a.score == null || a.sparse) && a.state !== 'error'; });
    if (waiting.length) pollTimer = setTimeout(() => { waiting.slice(0, 200).forEach((id) => queue(id, 0)); }, 6000);
    else pollTimer = setTimeout(refreshVisible, 30000);   // scores move as the crawl grows: refresh what is on screen
  }
  function refreshVisible() {
    [...badges.keys()].forEach((id) => { const set = badges.get(id); if ([...set].some((el) => el.isConnected)) queue(id, 1); });
  }
  function setOnline(ok, err) {
    if (online === ok) return;
    online = ok;
    if (!ok) toast('Trust Score: cannot reach the local server (' + (err && err.message || err) + '). Start it with scripts/serve_bluesky.py.');
  }
  let toastEl = null;
  function toast(msg) {
    if (!toastEl) { toastEl = document.createElement('div'); toastEl.className = 'ts-toast'; document.body.appendChild(toastEl); }
    toastEl.textContent = msg; toastEl.hidden = false;
    setTimeout(() => { toastEl.hidden = true; }, 6000);
  }

  // ---------- popover ----------
  let pop = null, popFor = null, hoverTimer = null;
  const fmt = (x, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function bar(c, max) {
    const w = max > 0 ? Math.min(50, 50 * Math.abs(c) / max) : 0;
    const left = c < 0 ? 50 - w : 50;
    return '<div class="ts-bar"><i style="left:' + left + '%;width:' + w + '%;background:' + (c > 0 ? 'var(--ts-green)' : 'var(--ts-red)') + '"></i></div>';
  }
  function html(a) {
    const k = cls(a);
    let h = '<div class="ts-brand"><b>' + SHIELD + ' Trust Score</b><span>Bluesky · add-on</span></div>';
    h += '<div class="ts-head"><div><div class="ts-handle">' + esc(a.displayName || a.handle || a.did) + '</div>';
    h += '<div class="ts-sub">@' + esc(a.handle || a.did) + '</div>';
    h += '<span class="ts-verdict ' + k + '">' + verdict(a) + '</span></div>';
    if (a.score != null) h += '<div class="ts-score"><div class="ts-big">' + Math.round(a.score * 100) + '<small>/100</small></div><div class="ts-sub">' + (a.sparse ? 'provisional · profile prior only' : 'trust = ½ − r') + '</div></div>';
    h += '</div>';
    if (a.score != null) {
      h += '<div class="ts-track"><div class="ts-marker" style="left:' + (100 * a.score) + '%"></div></div>';
      h += '<div class="ts-track-labels"><span>0 · bot-like</span><span>50</span><span>trusted · 100</span></div>';
    } else {
      h += '<div class="ts-state">' + (a.state === 'error' ? 'The crawler could not read this account.' : 'Queued for a breadth-first crawl of followers, follows and recent interactions; the score appears once its neighbourhood is in the graph.') + '</div>';
    }
    if (a.facts && a.facts.length) {
      h += '<div class="ts-section"><h4>Profile &amp; graph</h4><div class="ts-facts">';
      a.facts.forEach(([l, v]) => { h += '<div><span>' + esc(l) + '</span><b>' + esc(v) + '</b></div>'; });
      h += '</div></div>';
    }
    if (a.why && a.why.length) {
      const max = Math.max(...a.why.map((r) => Math.abs(r[1])), 1e-6);
      h += '<div class="ts-section"><h4>Why this score <span>← bot-like · trusted →</span></h4><div class="ts-rows">';
      a.why.forEach(([lab, c, note]) => {
        h += '<div class="ts-lab">' + esc(lab) + (note ? '<small>' + esc(note) + '</small>' : '') + '</div>' + bar(c, max) +
          '<div class="ts-val ' + (c > 0 ? 'ts-pos' : c < 0 ? 'ts-neg' : '') + '">' + (Math.abs(c) < 5e-4 ? '0' : fmt(c)) + '</div>';
      });
      const sum = a.why.reduce((s, r) => s + r[1], 0);
      h += '<div class="ts-lab ts-sum">Score = 0.50 + sum of the rows</div><div class="ts-sum"></div><div class="ts-val ts-sum">' + (0.5 + sum).toFixed(2) + '</div></div></div>';
    }
    if (a.neighbours && a.neighbours.length) {
      h += '<div class="ts-section"><h4>Most influential neighbours <span>their score → effect on this one</span></h4><table class="ts-nb">';
      a.neighbours.forEach(([n, rel, s, eff]) => {
        const nk = s >= 0.7 ? 'ts-green' : s >= 0.4 ? 'ts-amber' : 'ts-red';
        h += '<tr><td><span class="ts-dot ' + nk + '"></span>' + esc(n) + '</td><td class="ts-muted">' + esc(rel) + '</td><td class="ts-right">' + Math.round(s * 100) + '</td><td class="ts-right ' + (eff > 0 ? 'ts-pos' : 'ts-neg') + '">' + fmt(eff, 3) + '</td></tr>';
      });
      h += '</table></div>';
    }
    if (a.flags && a.flags.length) {
      h += '<div class="ts-section"><h4>Flags</h4><div class="ts-flags">';
      a.flags.forEach(([t, fk]) => { h += '<span class="ts-flag ' + (fk || '') + '">' + (fk === 'ts-red' ? BOLT : '') + esc(t) + '</span>'; });
      h += '</div></div>';
    }
    const seed = a.seed ? a.seed.label : null;
    h += '<div class="ts-section"><h4>Seed <span>your labels drive the propagation</span></h4><div class="ts-seedbar">' +
      '<button class="ts-btn' + (seed === 0 ? ' ts-on-green' : '') + '" data-ts-seed="0">Mark trusted</button>' +
      '<button class="ts-btn' + (seed === 1 ? ' ts-on-red' : '') + '" data-ts-seed="1">Mark untrusted</button>' +
      (seed != null ? '<button class="ts-btn" data-ts-seed="x">Remove seed</button>' : '') + '</div></div>';
    const when = a.computedAt ? new Date(a.computedAt * 1000).toLocaleString() : '—';
    h += '<div class="ts-foot"><span>Computed ' + when + ' · follow + interaction graph only, no post text</span><span><a data-ts-graph href="' + esc(dashUrl('?tab=network&hops=1&focus=' + encodeURIComponent(a.handle || a.did))) + '" target="_blank" rel="noopener">Open in graph ↗</a> · <a data-ts-how>How does this work?</a></span></div>';
    return h;
  }
  async function open(el, ident) {
    close();
    pop = document.createElement('div');
    pop.className = 'ts-pop' + (dark() ? ' ts-dark' : '');
    pop.innerHTML = html(accounts.get(ident) || { handle: ident, state: 'pending' });
    document.body.appendChild(pop);
    popFor = el; el.classList.add('ts-open');
    place(el);
    wirePop(ident);
    try {
      const full = await api('/bsky/explain/' + encodeURIComponent(ident));
      if (pop && popFor === el) { accounts.set(ident, full); repaint(ident); pop.innerHTML = html(full); place(el); wirePop(ident); }
    } catch (e) { /* keep the short version */ }
  }
  function place(el) {
    const r = el.getBoundingClientRect();
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = window.scrollX + r.left; let top = window.scrollY + r.bottom + 8;
    if (left + pw > window.scrollX + window.innerWidth - 8) left = window.scrollX + window.innerWidth - pw - 8;
    if (r.bottom + 8 + ph > window.innerHeight && r.top - 8 - ph > 0) top = window.scrollY + r.top - ph - 8;
    pop.style.left = Math.max(8, left) + 'px'; pop.style.top = top + 'px';
  }
  function wirePop(ident) {
    pop.addEventListener('mouseleave', () => { hoverTimer = setTimeout(() => { if (!popFor || !popFor.matches(':hover')) close(); }, 250); });
    pop.addEventListener('mouseenter', () => clearTimeout(hoverTimer));
    pop.querySelector('[data-ts-how]')?.addEventListener('click', showHow);
    pop.querySelectorAll('[data-ts-seed]').forEach((b) => b.addEventListener('click', async (e) => {
      e.preventDefault(); e.stopPropagation();
      const v = b.dataset.tsSeed;
      try {
        await api('/bsky/seeds', 'POST', v === 'x' ? { action: 'remove', ident } : { action: 'set', ident, label: Number(v) });
        toast('Seed updated — scores recomputed.');
        const full = await api('/bsky/explain/' + encodeURIComponent(ident));
        accounts.set(ident, full); repaint(ident);
        if (pop) { pop.innerHTML = html(full); wirePop(ident); }
        refreshVisible();
      } catch (err) { toast('Seed update failed: ' + err.message); }
    }));
  }
  function toggle(el, ident) { if (pop && popFor === el) close(); else open(el, ident); }
  function close() { if (pop) { pop.remove(); pop = null; } if (popFor) { popFor.classList.remove('ts-open'); popFor = null; } }

  const HOW = '<h3>How the Bluesky trust score is computed</h3>' +
    '<p>The add-on sends the accounts on the page to a local server. The server crawls the public Bluesky API <b>breadth-first</b>: for each account it reads the profile, up to 300 followers, up to 300 follows and the last 100 posts (replies, reposts, quotes, mentions), then enqueues the closest neighbours (mutual follows first) one level deeper. Everything is stored in a SQLite graph, so an account already crawled is never fetched again before its time-to-live expires.</p>' +
    '<p>On that graph the step-1 formula runs unchanged: <code>r = (1 − α) q + α P r</code>, <code>trust = ½ − r</code>. <code>q</code> is +½ for accounts you (or a moderation label) mark untrusted, −½ for accounts marked trusted, 0 otherwise; <code>P</code> averages the residual of the neighbours over typed, hub-damped channels (mutual follow, followed-by, follows, replies, reposts, quotes, mentions). No post text is used.</p>' +
    '<p>Every account also carries a transparent <b>profile prior</b> <code>q = λ (p − ½)</code>: account age, handle pattern, avatar, follower/follow ratio, posting rate and moderation labels, each with a documented weight (backend/trustscore/bluesky/local_prior.py). Seeds override it: by default a random draw of crawled accounts is marked trusted and moderation-labelled accounts untrusted; change any seed from the popover or the add-on menu. A grey "?" means the account has fewer than three stored edges: its score is the prior only and the crawl is queued. Rows in "Why this score" are the exact linear decomposition of the fixed point.</p>';
  function showHow() {
    close();
    let m = document.querySelector('.ts-modal-bg');
    if (!m) {
      m = document.createElement('div');
      m.className = 'ts-modal-bg' + (dark() ? ' ts-dark' : '');
      m.innerHTML = '<div class="ts-modal"><button class="ts-btn ts-close">Close</button>' + HOW + '</div>';
      document.body.appendChild(m);
      m.addEventListener('click', (e) => { if (e.target === m || e.target.classList.contains('ts-close')) m.hidden = true; });
    }
    m.hidden = false;
  }

  // ---------- wiring ----------
  function setEnabled(on) {
    enabled = on;
    document.body.classList.toggle('ts-off', !on);
    if (on) scan(); else close();
  }
  document.addEventListener('click', (e) => { if (pop && !pop.contains(e.target) && !(popFor && popFor.contains(e.target))) close(); }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { close(); const m = document.querySelector('.ts-modal-bg'); if (m) m.hidden = true; } });
  window.addEventListener('resize', () => { if (pop && popFor) place(popFor); });
  window.addEventListener('scroll', () => { clearTimeout(window.__tsScrollT); window.__tsScrollT = setTimeout(scan, 200); }, { passive: true });
  const mo = new MutationObserver(() => { clearTimeout(window.__tsMutT); window.__tsMutT = setTimeout(scan, 150); });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  if (hasRuntime) {
    chrome.storage.local.get(['enabled', 'server']).then((v) => { if (v.server) serverBase = v.server.replace(/\/$/, ''); if (v.enabled === false) setEnabled(false); });
    chrome.runtime.onMessage.addListener((m, _s, reply) => { if (m && m.type === 'visible') { reply(visibleIdents(20)); return true; } });
    chrome.storage.onChanged.addListener((ch) => {
      if (ch.enabled) setEnabled(ch.enabled.newValue !== false);
      if (ch.server && ch.server.newValue) serverBase = ch.server.newValue.replace(/\/$/, '');
      if (ch.seedsVersion) refreshVisible();
    });
  }
  window.__tsBluesky = { scan, accounts, refreshVisible, setEnabled, api, visibleIdents };
  scan();
})();
