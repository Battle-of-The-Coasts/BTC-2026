/* ============================================================
   Trust Score browser add-on — mock overlay logic.
   Pages call TrustOverlay.init({ platform, accounts, computedOn }).
   - every element with data-ts="<handle>" becomes a score badge
   - every element with data-ts-ring="<handle>" gets a coloured ring
   - clicking / hovering a badge opens the explanation popover
   - the banner switch hides everything the add-on injected
   ============================================================ */
(function () {
  const SHIELD = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5 4.5 5.5v6c0 5 3.3 8.6 7.5 10 4.2-1.4 7.5-5 7.5-10v-6z"/><path d="m9 12 2 2 4-4"/></svg>';
  const BOLT = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>';

  let cfg = null;
  let pop = null, popFor = null, hoverTimer = null;

  const fmt = (x, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
  const pct = (x) => Math.round(x * 100) + '%';

  function cls(a) {
    if (a.exempt) return 'ts-grey';
    if (a.score == null) return 'ts-grey';
    if (a.score >= 0.70) return 'ts-green';
    if (a.score >= 0.40) return 'ts-amber';
    return 'ts-red';
  }
  function verdict(a) {
    if (a.exempt) return a.exemptShort || 'not scored';
    if (a.score == null) return 'no score yet';
    if (a.score >= 0.70) return 'Likely genuine';
    if (a.score >= 0.40) return 'Uncertain';
    return 'Bot-like';
  }
  function badgeText(a) {
    if (a.exempt) return a.exemptShort || 'n/a';
    if (a.score == null) return '–';
    return Math.round(a.score * 100);
  }

  function renderBadges() {
    document.querySelectorAll('[data-ts]').forEach((el) => {
      const a = cfg.accounts[el.dataset.ts];
      if (!a) { el.remove(); return; }
      el.classList.add('ts-badge', cls(a));
      el.innerHTML = SHIELD + '<span class="ts-num">' + badgeText(a) + '</span>';
      el.setAttribute('role', 'button');
      el.setAttribute('tabindex', '0');
      el.title = 'Trust Score for ' + (a.handle || el.dataset.ts) + ' — click for the explanation';
      el.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); toggle(el, a); });
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(el, a); } });
      el.addEventListener('mouseenter', () => { clearTimeout(hoverTimer); hoverTimer = setTimeout(() => open(el, a), 260); });
      el.addEventListener('mouseleave', () => { clearTimeout(hoverTimer); hoverTimer = setTimeout(() => { if (!pop || !pop.matches(':hover')) close(); }, 220); });
    });
    document.querySelectorAll('[data-ts-ring]').forEach((el) => {
      const a = cfg.accounts[el.dataset.tsRing];
      if (a) el.classList.add('ts-ring', cls(a));
    });
  }

  function toggle(el, a) { if (pop && popFor === el) close(); else open(el, a); }

  function bar(c, max) {
    const w = max > 0 ? Math.min(50, 50 * Math.abs(c) / max) : 0;
    const left = c < 0 ? 50 - w : 50;
    const color = c > 0 ? 'var(--ts-green)' : 'var(--ts-red)';
    return '<div class="ts-bar"><i style="left:' + left + '%;width:' + w + '%;background:' + color + '"></i></div>';
  }

  function html(a) {
    const k = cls(a);
    let h = '<div class="ts-brand"><b>' + SHIELD + ' Trust Score</b><span>' + cfg.platformLabel + ' · add-on</span></div>';
    h += '<div class="ts-head"><div><div class="ts-handle">' + a.handle + '</div>';
    if (a.sub) h += '<div class="ts-sub">' + a.sub + '</div>';
    h += '<span class="ts-verdict ' + k + '">' + verdict(a) + '</span></div>';
    if (!a.exempt && a.score != null) {
      h += '<div class="ts-score"><div class="ts-big">' + Math.round(a.score * 100) + '<small>/100</small></div>';
      h += '<div class="ts-sub">P(bot) ≈ ' + pct(a.pbot) + '</div></div>';
    }
    h += '</div>';

    if (a.exempt) {
      h += '<div class="ts-section"><p class="ts-note" style="font-size:12.5px;color:var(--ts-ink)">' + a.exempt + '</p></div>';
    } else {
      h += '<div class="ts-track"><div class="ts-marker" style="left:' + (100 * a.score) + '%"></div></div>';
      h += '<div class="ts-track-labels"><span>0 · bot-like</span><span>50</span><span>trusted · 100</span></div>';

      if (a.facts && a.facts.length) {
        h += '<div class="ts-section"><h4>Profile &amp; activity</h4><div class="ts-facts">';
        a.facts.forEach(([l, v]) => { h += '<div><span>' + l + '</span><b>' + v + '</b></div>'; });
        h += '</div></div>';
      }

      if (a.why && a.why.length) {
        const max = Math.max(...a.why.map((r) => Math.abs(r[1])), 1e-6);
        h += '<div class="ts-section"><h4>Why this score <span>← bot-like · trusted →</span></h4><div class="ts-rows">';
        a.why.forEach(([lab, c, note]) => {
          h += '<div class="ts-lab">' + lab + (note ? '<small>' + note + '</small>' : '') + '</div>' + bar(c, max) +
               '<div class="ts-val ' + (c > 0 ? 'ts-pos' : c < 0 ? 'ts-neg' : '') + '">' + (c === 0 ? '0' : fmt(c)) + '</div>';
        });
        const sum = a.why.reduce((s, r) => s + r[1], 0);
        h += '<div class="ts-lab ts-sum">Score = 0.50 + sum of the rows</div><div class="ts-sum"></div><div class="ts-val ts-sum">' + (0.5 + sum).toFixed(2) + '</div>';
        h += '</div></div>';
      }

      if (a.neighbours && a.neighbours.length) {
        h += '<div class="ts-section"><h4>Most influential neighbours <span>their score → effect on this one</span></h4><table class="ts-nb">';
        a.neighbours.forEach(([n, rel, s, eff]) => {
          const nk = s >= 0.7 ? 'ts-green' : s >= 0.4 ? 'ts-amber' : 'ts-red';
          h += '<tr><td><span class="ts-dot ' + nk + '"></span>' + n + '</td><td class="ts-muted">' + rel + '</td>' +
               '<td class="ts-right">' + Math.round(s * 100) + '</td><td class="ts-right ' + (eff > 0 ? 'ts-pos' : 'ts-neg') + '">' + fmt(eff, 3) + '</td></tr>';
        });
        h += '</table></div>';
      }

      if (a.flags && a.flags.length) {
        h += '<div class="ts-section"><h4>Flags</h4><div class="ts-flags">';
        a.flags.forEach(([t, fk]) => { h += '<span class="ts-flag ' + (fk || '') + '">' + (fk === 'ts-red' ? BOLT : '') + t + '</span>'; });
        h += '</div></div>';
      }
      if (a.note) h += '<p class="ts-note">' + a.note + '</p>';
    }

    h += '<div class="ts-foot"><span>Computed ' + cfg.computedOn + ' · profile + network only, no post text</span><span><a data-ts-how>How does this work?</a> · <a>Report</a></span></div>';
    return h;
  }

  function open(el, a) {
    close();
    pop = document.createElement('div');
    pop.className = 'ts-pop' + (cfg.dark ? ' ts-dark' : '');
    pop.innerHTML = html(a);
    document.body.appendChild(pop);
    popFor = el; el.classList.add('ts-open');
    const r = el.getBoundingClientRect();
    const pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = window.scrollX + r.left; let top = window.scrollY + r.bottom + 8;
    if (left + pw > window.scrollX + window.innerWidth - 8) left = window.scrollX + window.innerWidth - pw - 8;
    if (r.bottom + 8 + ph > window.innerHeight && r.top - 8 - ph > 0) top = window.scrollY + r.top - ph - 8;
    pop.style.left = left + 'px'; pop.style.top = top + 'px';
    pop.addEventListener('mouseleave', () => { hoverTimer = setTimeout(() => { if (!el.matches(':hover')) close(); }, 220); });
    pop.addEventListener('mouseenter', () => clearTimeout(hoverTimer));
    pop.querySelector('[data-ts-how]')?.addEventListener('click', showHow);
  }
  function close() {
    if (pop) { pop.remove(); pop = null; }
    if (popFor) { popFor.classList.remove('ts-open'); popFor = null; }
  }

  function showHow() {
    close();
    let m = document.querySelector('.ts-modal-bg');
    if (!m) {
      m = document.createElement('div');
      m.className = 'ts-modal-bg' + (cfg.dark ? ' ts-dark' : '');
      m.innerHTML = '<div class="ts-modal"><button class="ts-btn ts-close">Close</button>' + cfg.how + '</div>';
      document.body.appendChild(m);
      m.addEventListener('click', (e) => { if (e.target === m || e.target.classList.contains('ts-close')) m.hidden = true; });
    }
    m.hidden = false;
  }

  function wireBanner() {
    document.querySelectorAll('.ts-switch').forEach((s) => s.addEventListener('click', () => {
      document.body.classList.toggle('ts-off'); close();
      const on = !document.body.classList.contains('ts-off');
      s.querySelector('span').textContent = on ? 'Overlay ON' : 'Overlay OFF';
    }));
    document.querySelectorAll('[data-ts-guide]').forEach((b) => b.addEventListener('click', () => {
      const g = document.querySelector('.ts-guide'); g.hidden = !g.hidden;
      b.textContent = g.hidden ? 'What am I looking at?' : 'Hide guide';
    }));
    document.querySelectorAll('[data-ts-how]').forEach((b) => b.addEventListener('click', showHow));
    document.querySelectorAll('[data-ts-jump]').forEach((b) => b.addEventListener('click', () => {
      const t = document.getElementById(b.dataset.tsJump); if (!t) return;
      t.scrollIntoView({ behavior: 'smooth', block: 'center' });
      t.classList.remove('ts-pulse'); void t.offsetWidth; t.classList.add('ts-pulse');
      const badge = t.querySelector('[data-ts]');
      if (badge && b.dataset.tsOpen !== 'no') setTimeout(() => open(badge, cfg.accounts[badge.dataset.ts]), 700);
    }));
    document.addEventListener('click', (e) => { if (pop && !pop.contains(e.target)) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { close(); const m = document.querySelector('.ts-modal-bg'); if (m) m.hidden = true; } });
    window.addEventListener('resize', () => {           // keep the popover glued to its badge
      if (!pop || !popFor) return;
      const r = popFor.getBoundingClientRect(), pw = pop.offsetWidth;
      let left = window.scrollX + r.left;
      if (left + pw > window.scrollX + window.innerWidth - 8) left = window.scrollX + window.innerWidth - pw - 8;
      pop.style.left = left + 'px';
    });
  }

  window.TrustOverlay = {
    init(c) { cfg = c; renderBadges(); wireBanner(); },
    SHIELD,
  };
})();
