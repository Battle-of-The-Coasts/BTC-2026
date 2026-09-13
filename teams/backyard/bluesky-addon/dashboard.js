/* Trust Score · Bluesky dashboard. Static page served by the add-on API (same origin), d3 for charts + canvas graph. */
(function () {
  const API = location.origin;
  const $ = (s, el) => (el || document).querySelector(s);
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmtN = (x) => x == null ? '–' : Number(x).toLocaleString();
  const band = (t) => t == null ? 'grey' : t >= 0.7 ? 'green' : t >= 0.4 ? 'amber' : 'red';
  const verdict = (t) => t == null ? 'no score' : t >= 0.7 ? 'Likely genuine' : t >= 0.4 ? 'Uncertain' : 'Bot-like';
  const trustColor = d3.interpolateRgbBasis(['#e5484d', '#f5b731', '#30a46c']);
  const EDGE_COLORS = { follow: '#3d4a66', reply: '#4cc9f0', repost: '#b58cff', quote: '#f5b731', mention: '#30a46c' };
  const DEPTH_COLORS = ['#4cc9f0', '#b58cff', '#f5b731', '#9a9fb3'];
  const api = async (path, method, body) => {
    const r = await fetch(API + path, { method: method || 'GET', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
    if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + path);
    return r.json();
  };
  const params = new URLSearchParams(location.search);
  const state = { tab: params.get('tab') || (params.get('focus') ? 'network' : 'overview'), focus: params.get('focus') || '', hops: Number(params.get('hops') || 1),
    maxNodes: Number(params.get('max_nodes') || 800), color: 'trust', contrast: 6, edgeOn: { follow: true, reply: true, repost: true, quote: true, mention: true }, graph: null, selected: null };
  const content = $('#content'), drawer = $('#drawer');

  // ---------------- tabs ----------------
  function setTab(t) {
    state.tab = t;
    document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === t));
    content.className = 'content' + (t === 'network' ? ' net' : '');
    const u = new URL(location); u.searchParams.set('tab', t); history.replaceState(null, '', u);
    ({ overview: renderOverview, network: renderNetwork, accounts: renderAccounts, seeds: renderSeeds, method: renderMethod })[t]();
  }
  document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));

  // ---------------- charts ----------------
  function barChart(el, data, series, opts) {
    opts = opts || {};
    const W = el.clientWidth || 500, H = el.clientHeight || 190, m = { t: 8, r: 8, b: 26, l: 44 };
    const svg = d3.select(el).html('').append('svg').attr('width', W).attr('height', H);
    const x0 = d3.scaleBand().domain(data.map((d) => d.label)).range([m.l, W - m.r]).padding(0.15);
    const x1 = d3.scaleBand().domain(series.map((s) => s.key)).range([0, x0.bandwidth()]).padding(0.05);
    const maxY = d3.max(data, (d) => d3.max(series, (s) => d[s.key])) || 1;
    const y = (opts.sqrt ? d3.scaleSqrt() : d3.scaleLinear()).domain([0, maxY]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x0).tickSize(0)).selectAll('text').style('font-size', data.length > 12 ? '9px' : '11px');
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('~s')));
    const g = svg.append('g').selectAll('g').data(data).join('g').attr('transform', (d) => `translate(${x0(d.label)},0)`);
    g.selectAll('rect').data((d) => series.map((s) => ({ key: s.key, v: d[s.key], color: typeof s.color === 'function' ? s.color(d) : s.color, d })))
      .join('rect').attr('x', (d) => x1(d.key)).attr('width', x1.bandwidth()).attr('y', (d) => y(d.v)).attr('height', (d) => y(0) - y(d.v)).attr('fill', (d) => d.color).attr('rx', 2)
      .append('title').text((d) => `${d.d.label} · ${d.key}: ${fmtN(d.v)}`);
    if (opts.sqrt) svg.append('text').attr('x', W - m.r).attr('y', m.t + 8).attr('text-anchor', 'end').text('√ scale');
  }
  function lineChart(el, data, key, color, opts) {
    opts = opts || {};
    const W = el.clientWidth || 500, H = el.clientHeight || 190, m = { t: 8, r: 8, b: 26, l: 48 };
    const svg = d3.select(el).html('').append('svg').attr('width', W).attr('height', H);
    const x = d3.scaleLinear().domain(d3.extent(data, (d) => d.run_id)).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d[key]) || 1]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('d')));
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('~s')));
    svg.append('path').datum(data).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('d', d3.line().x((d) => x(d.run_id)).y((d) => y(d[key])));
    svg.append('text').attr('x', W - m.r).attr('y', m.t + 8).attr('text-anchor', 'end').text(opts.label || key);
  }

  // ---------------- overview ----------------
  async function renderOverview() {
    content.innerHTML = '<div class="loading">Loading analytics…</div>';
    let a;
    try { a = await api('/bsky/analytics'); } catch (e) { content.innerHTML = `<div class="error">${esc(e.message)}. Is scripts/serve_bluesky.py running?</div>`; return; }
    const s = a.store, c = a.crawler, r = a.last_run;
    $('#topinfo').textContent = `${fmtN(s.actors)} nodes · ${fmtN(a.n_edges_total)} edges · ${fmtN(s.crawled)} crawled · ${fmtN(s.seeds)} seeds`;
    const b = a.scores.bands;
    const bandRow = (o) => ['green', 'amber', 'red', 'grey'].map((k) => `<span class="pill ${k}">${k === 'grey' ? 'n/a' : k === 'green' ? '≥70' : k === 'amber' ? '40–69' : '<40'} · ${fmtN(o[k] || 0)}</span>`).join(' ');
    content.innerHTML = `
      <div class="grid cols-5">
        <div class="card"><div class="title">Accounts in graph</div><div class="kpi">${fmtN(s.actors)}<small>${fmtN(s.crawled)} crawled (own lists read)</small></div></div>
        <div class="card"><div class="title">Edges</div><div class="kpi">${fmtN(s.follows)}<small>follows</small></div><div class="small muted">${fmtN(s.interactions)} interaction events</div></div>
        <div class="card"><div class="title">Seeds</div><div class="kpi">${fmtN(s.seeds)}<small>${a.seeds_by.map((x) => `${x.n} ${x.source} ${x.label ? 'untrusted' : 'trusted'}`).join(' · ')}</small></div></div>
        <div class="card"><div class="title">Last propagation</div><div class="kpi">${r ? r.iterations + '<small>iterations · ' + (r.converged ? 'converged' : 'not converged') + '</small>' : '–'}</div><div class="small muted">${r ? new Date(r.computed_at * 1000).toLocaleString() + ' · run #' + r.run_id : 'no run yet'}</div></div>
        <div class="card"><div class="title">Crawler</div><div class="kpi">${fmtN(c.queue.total)}<small>queued</small></div><div class="small muted">${c.current ? 'working · ' : 'idle · '}${fmtN(c.api_calls)} API calls · ${c.errors} errors</div></div>
      </div>
      <div class="grid cols-2">
        <div class="card"><h2>Trust score distribution (½ − r)</h2><div class="small muted" style="margin-bottom:6px">all nodes vs. crawled nodes · 20 bins</div><div class="chart" id="ch-hist"></div>
          <div class="row small" style="margin-top:8px"><b>All:</b> ${bandRow(b.all)}</div><div class="row small" style="margin-top:4px"><b>Crawled:</b> ${bandRow(b.crawled)}</div></div>
        <div class="card"><h2>Degree distribution</h2><div class="small muted" style="margin-bottom:6px">edges per node in the stored graph (follows + interactions)</div><div class="chart" id="ch-deg"></div></div>
      </div>
      <div class="grid cols-3">
        <div class="card"><h2>Graph growth</h2><div class="chart" id="ch-runs-nodes"></div><div class="chart" id="ch-runs-edges" style="margin-top:6px"></div></div>
        <div class="card"><h2>Crawl depth</h2><table class="tbl"><tr><th>depth</th><th>nodes</th><th>crawled</th></tr>${a.depth.map((d) => `<tr><td>${d.depth == null ? '?' : d.depth}</td><td class="num">${fmtN(d.n)}</td><td class="num">${fmtN(d.crawled)}</td></tr>`).join('')}</table>
          <h2 style="margin-top:14px">Channels</h2><table class="tbl"><tr><th>channel</th><th>edges</th><th>weight</th></tr>${a.channels.map((x) => `<tr><td>${esc(x.name)}<div class="small muted">${esc(x.note)}</div></td><td class="num">${fmtN(x.edges)}</td><td class="num">${esc(x.weight)}</td></tr>`).join('')}</table></div>
        <div class="card"><h2>Extremes among crawled accounts</h2>
          <h3>Most trusted</h3><table class="tbl">${a.top.trusted.map(topRow).join('')}</table>
          <h3 style="margin-top:10px">Least trusted</h3><table class="tbl">${a.top.untrusted.map(topRow).join('')}</table></div>
      </div>
      <div class="card"><h2>Parameters</h2><div class="row small muted"><span>α = ${a.scoring.alpha}</span><span>seed residual = ${a.scoring.seed_residual}</span><span>hub damping = ${a.scoring.hub_damping}</span><span>random seeds = ${a.scoring.random_seed_count}</span>
        <span>max depth = ${c.settings.max_depth}</span><span>expand/node = ${c.settings.expand_per_node}</span><span>list pages = ${c.settings.list_pages}</span><span>TTL = ${c.settings.ttl_hours} h</span><span>${c.settings.requests_per_second} req/s</span></div></div>`;
    content.querySelectorAll('tr.clickable').forEach((tr) => tr.addEventListener('click', () => openNode(tr.dataset.did)));
    barChart($('#ch-hist'), a.scores.hist.map((h) => ({ label: Math.round(h.lo * 100) + '', all: h.all, crawled: h.crawled, lo: h.lo })), [{ key: 'all', color: (d) => trustColor(d.lo + 0.025) }, { key: 'crawled', color: '#4cc9f0' }], { sqrt: true });
    barChart($('#ch-deg'), a.degrees, [{ key: 'all', color: '#3d4a66' }, { key: 'crawled', color: '#4cc9f0' }], { sqrt: true });
    if (a.runs.length > 1) { lineChart($('#ch-runs-nodes'), a.runs, 'n_nodes', '#4cc9f0', { label: 'nodes per run' }); lineChart($('#ch-runs-edges'), a.runs, 'n_edges', '#b58cff', { label: 'edges per run' }); }
  }
  const topRow = (x) => `<tr class="clickable" data-did="${esc(x.did)}"><td>@${esc(x.handle || x.did)}${x.seed != null ? ` <span class="pill ${x.seed ? 'red' : 'green'}">seed</span>` : ''}</td><td class="num">${fmtN(x.deg)} edges</td><td class="num"><span class="pill ${band(x.trust)}">${Math.round(x.trust * 100)}</span></td></tr>`;

  // ---------------- network ----------------
  let net = null;   // { canvas, ctx, nodes, edges, byId, zoom, transform, quadtree }
  async function renderNetwork() {
    content.innerHTML = `<div class="netwrap">
      <div class="toolbar">
        <input id="focus" placeholder="focus @handle(s), comma-separated" value="${esc(state.focus)}" style="width:260px">
        <label class="small muted">hops <select id="hops">${[0, 1, 2, 3].map((h) => `<option ${h === state.hops ? 'selected' : ''}>${h}</option>`).join('')}</select></label>
        <label class="small muted">max nodes <select id="maxn">${[300, 800, 1500, 3000].map((h) => `<option ${h === state.maxNodes ? 'selected' : ''}>${h}</option>`).join('')}</select></label>
        <button class="btn" id="load">Load</button><button class="btn" id="core">Core graph</button><button class="btn" id="fit" title="fit the whole graph in the view">Fit</button>
        <label class="small muted">colour <select id="color"><option value="trust">trust</option><option value="depth">crawl depth</option><option value="seed">seeds</option><option value="crawled">crawled vs. leaf</option></select></label>
        <div class="slider">contrast ×<span id="cval">${state.contrast}</span><input type="range" id="contrast" min="1" max="20" value="${state.contrast}"></div>
        <span class="small muted">edges:</span>${Object.keys(EDGE_COLORS).map((k) => `<label class="small"><input type="checkbox" data-edge="${k}" ${state.edgeOn[k] ? 'checked' : ''}> <span style="color:${EDGE_COLORS[k]}">■</span> ${k}</label>`).join('')}
        <input id="search" placeholder="find in graph" style="width:140px">
        <span class="small muted" id="netinfo"></span>
      </div>
      <div class="netcanvas"><canvas id="cv"></canvas><div class="hint">scroll to zoom · drag to pan · click a node for the explanation</div><div class="tip" id="tip"></div>
        <div class="legend" id="netlegend"></div></div></div>`;
    $('#load').addEventListener('click', () => { state.focus = $('#focus').value.trim(); state.hops = Number($('#hops').value); state.maxNodes = Number($('#maxn').value); loadGraph(); });
    $('#focus').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('#load').click(); });
    $('#core').addEventListener('click', () => { state.focus = ''; $('#focus').value = ''; loadGraph(); });
    $('#fit').addEventListener('click', () => fitView());
    $('#color').value = state.color; $('#color').addEventListener('change', () => { state.color = $('#color').value; draw(); legend(); });
    $('#contrast').addEventListener('input', () => { state.contrast = Number($('#contrast').value); $('#cval').textContent = state.contrast; draw(); });
    content.querySelectorAll('[data-edge]').forEach((cb) => cb.addEventListener('change', () => { state.edgeOn[cb.dataset.edge] = cb.checked; draw(); }));
    $('#search').addEventListener('keydown', (e) => { if (e.key !== 'Enter' || !net) return; const q = $('#search').value.toLowerCase().replace(/^@/, ''); const n = net.nodes.find((x) => (x.h || '').toLowerCase().includes(q) || (x.n || '').toLowerCase().includes(q)); if (n) { select(n.id); centerOn(n); } });
    if (state.graph) { setupCanvas(); } else { loadGraph(); }
  }
  async function loadGraph() {
    $('#netinfo').textContent = 'loading…';
    const u = new URL(location); u.searchParams.set('focus', state.focus); u.searchParams.set('hops', state.hops); u.searchParams.set('max_nodes', state.maxNodes); history.replaceState(null, '', u);
    try { state.graph = await api(`/bsky/graph?focus=${encodeURIComponent(state.focus)}&hops=${state.hops}&max_nodes=${state.maxNodes}`); }
    catch (e) { $('#netinfo').textContent = e.message; return; }
    setupCanvas();
  }
  function setupCanvas() {
    const g = state.graph;
    const canvas = $('#cv'), box = canvas.parentElement;
    const W = box.clientWidth, H = box.clientHeight, dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext('2d');
    const nodes = g.nodes.map((n) => ({ ...n, r: 2 + Math.sqrt(n.deg || 1) * 0.6 }));
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const edges = g.edges.filter((e) => byId.has(e.s) && byId.has(e.t)).map((e) => ({ ...e, source: e.s, target: e.t }));
    $('#netinfo').textContent = `${g.mode === 'focus' ? 'ego graph, ' + g.hops + ' hop(s)' : 'core: crawled accounts + best-connected neighbours'} · ${fmtN(nodes.length)} nodes · ${fmtN(edges.length)} edges${g.truncated ? ' · truncated to max nodes' : ''} · layout…`;
    // static force layout (like the precomputed x/y of the dataset dashboard), then interactive canvas
    const sim = d3.forceSimulation(nodes).force('link', d3.forceLink(edges).id((d) => d.id).distance(18).strength((e) => e.k === 'follow' ? 0.15 : 0.4))
      .force('charge', d3.forceManyBody().strength(-14).theta(0.9).distanceMax(400)).force('center', d3.forceCenter(0, 0)).force('collide', d3.forceCollide((d) => d.r + 0.5).iterations(1)).stop();
    const ticks = nodes.length > 1500 ? 120 : nodes.length > 600 ? 200 : 300;
    let i = 0;
    const job = (setupCanvas.job = (setupCanvas.job || 0) + 1);
    // timer-based chunks (animation frames stop in background tabs, timers do not)
    const step = () => {
      if (setupCanvas.job !== job) return;
      const t0 = performance.now();
      while (i < ticks && performance.now() - t0 < 40) { sim.tick(); i++; }
      draw();
      if (i < ticks) { setTimeout(step, 0); return; }
      fitView();                       // the initial scale is a guess from the node count; fit the real extent
      const el = $('#netinfo'); if (el) el.textContent = el.textContent.replace(' · layout…', '');
    };
    net = { canvas, ctx, nodes, edges, byId, W, H, dpr, transform: d3.zoomIdentity, focusIds: new Set(g.focus), zoom: null };
    const ext = 0.45 * Math.min(W, H) / Math.sqrt(nodes.length) * 4;
    net.transform = d3.zoomIdentity.translate(W / 2, H / 2).scale(Math.max(0.2, Math.min(2.5, ext / 40)));
    const zoom = d3.zoom().scaleExtent([0.05, 12]).on('zoom', (ev) => { net.transform = ev.transform; draw(); });
    net.zoom = zoom;
    d3.select(canvas).call(zoom).call(zoom.transform, net.transform);
    canvas.onmousemove = (ev) => { const n = hit(ev); const tip = $('#tip'); if (n) { tip.style.display = 'block'; tip.style.left = (ev.offsetX + 12) + 'px'; tip.style.top = (ev.offsetY + 12) + 'px'; tip.innerHTML = `<b>${esc(n.n || '')}</b> @${esc(n.h || n.id)}<br>${n.t == null ? 'no score' : 'trust ' + Math.round(n.t * 100)} · ${n.deg} edges · depth ${n.d == null ? '?' : n.d}${n.c ? ' · crawled' : ''}${n.s != null ? ' · ' + (n.s ? 'untrusted' : 'trusted') + ' seed' : ''}`; canvas.style.cursor = 'pointer'; } else { tip.style.display = 'none'; canvas.style.cursor = 'grab'; } };
    canvas.onclick = (ev) => { const n = hit(ev); if (n) { select(n.id); } };
    setTimeout(step, 0);
    legend();
  }
  function hit(ev) {
    if (!net) return null;
    const t = net.transform; const x = (ev.offsetX - t.x) / t.k, y = (ev.offsetY - t.y) / t.k;
    let best = null, bd = Infinity;
    for (const n of net.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2; const rr = (n.r + 3 / t.k) ** 2; if (d < rr && d < bd) { best = n; bd = d; } }
    return best;
  }
  function nodeColor(n) {
    if (state.color === 'depth') return DEPTH_COLORS[Math.min(n.d == null ? 3 : n.d, 3)];
    if (state.color === 'seed') return n.s == null ? '#3d4a66' : n.s ? '#e5484d' : '#30a46c';
    if (state.color === 'crawled') return n.c ? '#4cc9f0' : '#3d4a66';
    if (n.t == null) return '#6f7285';
    return trustColor(Math.max(0, Math.min(1, 0.5 + (n.t - 0.5) * state.contrast)));
  }
  function draw() {
    if (!net) return;
    const { ctx, nodes, edges, W, H, dpr, transform: t } = net;
    ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H); ctx.translate(t.x, t.y); ctx.scale(t.k, t.k);
    const sel = state.selected;
    ctx.lineWidth = 0.6 / t.k;
    for (const e of edges) {
      if (!state.edgeOn[e.k]) continue;
      const a = net.byId.get(e.s), b = net.byId.get(e.t); if (!a || !b) continue;
      const hi = sel && (e.s === sel || e.t === sel);
      ctx.strokeStyle = hi ? '#8fa3ff' : EDGE_COLORS[e.k]; ctx.globalAlpha = hi ? 0.9 : e.k === 'follow' ? 0.18 : 0.5; ctx.lineWidth = (hi ? 1.6 : 0.6) / t.k;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (const n of nodes) {
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI); ctx.fillStyle = nodeColor(n); ctx.fill();
      if (n.s != null || n.f || n.id === sel) { ctx.lineWidth = (n.id === sel ? 2.2 : 1.2) / t.k; ctx.strokeStyle = n.id === sel ? '#fff' : n.f ? '#4cc9f0' : n.s ? '#e5484d' : '#30a46c'; ctx.stroke(); }
    }
    ctx.fillStyle = '#c9ccd8'; ctx.font = `${11 / t.k}px Inter, system-ui, sans-serif`;
    for (const n of nodes) {
      const show = n.id === sel || n.f || (t.k > 1.2 && n.r > 6) || (t.k > 2.2 && (n.c || n.r > 4)) || t.k > 4;
      if (show && n.h) ctx.fillText('@' + n.h, n.x + n.r + 2 / t.k, n.y + 3 / t.k);
    }
    ctx.restore();
  }
  function legend() {
    const el = $('#netlegend'); if (!el) return;
    let h = '';
    if (state.color === 'trust') h = `<span><span class="sw" style="background:${trustColor(0)}"></span>bot-like</span><span><span class="sw" style="background:${trustColor(0.5)}"></span>50</span><span><span class="sw" style="background:${trustColor(1)}"></span>trusted</span><span><span class="sw" style="background:#6f7285"></span>no score</span><span class="muted">contrast stretches deviations from 50 for visibility</span>`;
    else if (state.color === 'depth') h = DEPTH_COLORS.map((c, i) => `<span><span class="sw" style="background:${c}"></span>depth ${i === 3 ? '3+/?' : i}</span>`).join('');
    else if (state.color === 'seed') h = `<span><span class="sw" style="background:#30a46c"></span>trusted seed</span><span><span class="sw" style="background:#e5484d"></span>untrusted seed</span><span><span class="sw" style="background:#3d4a66"></span>other</span>`;
    else h = `<span><span class="sw" style="background:#4cc9f0"></span>crawled (own lists read)</span><span><span class="sw" style="background:#3d4a66"></span>leaf (seen in a list only)</span>`;
    h += `<span style="margin-left:auto">ring: <span style="color:#4cc9f0">focus</span> · <span style="color:#30a46c">trusted seed</span> · <span style="color:#e5484d">untrusted seed</span></span>`;
    el.innerHTML = h;
  }
  function fitView(pad = 44) {
    if (!net || !net.nodes.length) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of net.nodes) {
      if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) continue;
      x0 = Math.min(x0, n.x - n.r); x1 = Math.max(x1, n.x + n.r);
      y0 = Math.min(y0, n.y - n.r); y1 = Math.max(y1, n.y + n.r);
    }
    if (!Number.isFinite(x0)) return;
    const k = Math.max(0.05, Math.min(12, Math.min((net.W - 2 * pad) / Math.max(1, x1 - x0), (net.H - 2 * pad) / Math.max(1, y1 - y0))));
    net.transform = d3.zoomIdentity.translate(net.W / 2 - k * (x0 + x1) / 2, net.H / 2 - k * (y0 + y1) / 2).scale(k);
    if (net.zoom) d3.select(net.canvas).call(net.zoom.transform, net.transform);
    draw();
  }

  function centerOn(n) { if (!net) return; net.transform = d3.zoomIdentity.translate(net.W / 2 - n.x * 2.5, net.H / 2 - n.y * 2.5).scale(2.5); d3.select(net.canvas).call(d3.zoom().transform, net.transform); draw(); }
  function select(id) { state.selected = id; draw(); openNode(id); }

  // ---------------- node drawer ----------------
  async function openNode(ident) {
    drawer.hidden = false; drawer.innerHTML = '<div class="loading">Loading…</div>';
    let a;
    try { a = await api('/bsky/explain/' + encodeURIComponent(ident)); } catch (e) { drawer.innerHTML = `<div class="error">${esc(e.message)}</div>`; return; }
    state.selected = a.did; if (net) draw();
    const k = band(a.score);
    let h = `<div class="row" style="justify-content:space-between"><h2 style="margin:0">${esc(a.displayName || a.handle || a.did)}</h2><button class="btn small" id="dclose">Close</button></div>
      <div class="small muted">@${esc(a.handle || a.did)} · <a href="https://bsky.app/profile/${esc(a.handle || a.did)}" target="_blank">open on bsky.app</a> · <a href="#" id="dfocus">ego graph</a></div>
      <div class="row" style="margin-top:8px"><span class="pill ${k}">${verdict(a.score)}</span><span class="kpi">${a.score == null ? '–' : Math.round(a.score * 100)}<small>/100</small></span><span class="small muted">${a.state}${a.depth != null ? ' · depth ' + a.depth : ''}</span></div>`;
    if (a.score != null) h += `<div class="trustbar"><div class="marker" style="left:${a.score * 100}%"></div></div>`;
    if (a.facts && a.facts.length) h += `<h3 style="margin-top:12px">Profile & graph</h3><div class="facts">${a.facts.map(([l, v]) => `<div><span>${esc(l)}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
    if (a.why && a.why.length) {
      const max = Math.max(...a.why.map((r) => Math.abs(r[1])), 1e-6);
      h += `<h3 style="margin-top:12px">Why this score</h3><div class="contrib">${a.why.map(([lab, c, note]) => { const w = Math.min(50, 50 * Math.abs(c) / max), left = c < 0 ? 50 - w : 50; return `<div>${esc(lab)}<div class="small muted">${esc(note || '')}</div></div><div class="bar"><i style="left:${left}%;width:${w}%;background:${c > 0 ? '#30a46c' : '#e5484d'}"></i></div><div class="num ${c > 0 ? 'pos' : c < 0 ? 'neg' : ''}">${(c >= 0 ? '+' : '−') + Math.abs(c).toFixed(3)}</div>`; }).join('')}
        <div class="muted small">Score = 0.50 + rows</div><div></div><div><b>${(0.5 + a.why.reduce((s, r) => s + r[1], 0)).toFixed(2)}</b></div></div>`;
    }
    if (a.neighbours && a.neighbours.length) h += `<h3 style="margin-top:12px">Most influential neighbours</h3><table class="tbl">${a.neighbours.map(([n, rel, s, eff, did]) => `<tr class="clickable" data-did="${esc(did)}"><td>@${esc(n)}</td><td class="muted">${esc(rel)}</td><td class="num"><span class="pill ${band(s)}">${Math.round(s * 100)}</span></td><td class="num ${eff > 0 ? 'pos' : 'neg'}">${(eff >= 0 ? '+' : '−') + Math.abs(eff).toFixed(3)}</td></tr>`).join('')}</table>`;
    if (a.flags && a.flags.length) h += `<h3 style="margin-top:12px">Flags</h3><div>${a.flags.map(([t, c]) => `<span class="flag ${(c || '').replace('ts-', '')}">${esc(t)}</span>`).join('')}</div>`;
    const seed = a.seed ? a.seed.label : null;
    h += `<h3 style="margin-top:12px">Seed</h3><div class="row"><button class="btn small ${seed === 0 ? 'green' : ''}" data-seed="0">Mark trusted</button><button class="btn small ${seed === 1 ? 'red' : ''}" data-seed="1">Mark untrusted</button>${seed != null ? '<button class="btn small" data-seed="x">Remove seed</button>' : ''}</div>`;
    drawer.innerHTML = h;
    $('#dclose').addEventListener('click', () => { drawer.hidden = true; state.selected = null; if (net) draw(); });
    $('#dfocus').addEventListener('click', (e) => { e.preventDefault(); state.focus = a.handle || a.did; state.hops = Math.max(1, state.hops); if (state.tab !== 'network') setTab('network'); else { $('#focus').value = state.focus; loadGraph(); } });
    drawer.querySelectorAll('tr.clickable').forEach((tr) => tr.addEventListener('click', () => { if (net && net.byId.has(tr.dataset.did)) { select(tr.dataset.did); centerOn(net.byId.get(tr.dataset.did)); } else openNode(tr.dataset.did); }));
    drawer.querySelectorAll('[data-seed]').forEach((b) => b.addEventListener('click', async () => {
      const v = b.dataset.seed;
      try { await api('/bsky/seeds', 'POST', v === 'x' ? { action: 'remove', ident: a.did } : { action: 'set', ident: a.did, label: Number(v) }); }
      catch (e) { alert(e.message); return; }
      if (state.graph) { try { state.graph = await api(`/bsky/graph?focus=${encodeURIComponent(state.focus)}&hops=${state.hops}&max_nodes=${state.maxNodes}`); if (net) { const pos = new Map(net.nodes.map((n) => [n.id, n])); state.graph.nodes.forEach((n) => { const p = pos.get(n.id); if (p) { n.x = p.x; n.y = p.y; } }); net.nodes.forEach((n) => { const u = state.graph.nodes.find((x) => x.id === n.id); if (u) { n.t = u.t; n.s = u.s; } }); draw(); } } catch (e) { /* ignore */ } }
      openNode(a.did);
    }));
  }

  // ---------------- accounts ----------------
  const acc = { q: '', sort: 'trust', desc: true, offset: 0, limit: 100, all: false };
  async function renderAccounts() {
    content.innerHTML = `<div class="row" style="margin-bottom:10px"><input id="aq" placeholder="search handle or name" value="${esc(acc.q)}" style="width:240px"><label class="small muted"><input type="checkbox" id="aall" ${acc.all ? 'checked' : ''}> include leaves (not crawled)</label><span class="small muted" id="ainfo"></span></div><div id="atbl"></div>`;
    $('#aq').addEventListener('keydown', (e) => { if (e.key === 'Enter') { acc.q = $('#aq').value.trim(); acc.offset = 0; loadAccounts(); } });
    $('#aall').addEventListener('change', () => { acc.all = $('#aall').checked; acc.offset = 0; loadAccounts(); });
    loadAccounts();
  }
  async function loadAccounts() {
    const d = await api(`/bsky/accounts?q=${encodeURIComponent(acc.q)}&sort=${acc.sort}&desc=${acc.desc ? 1 : 0}&limit=${acc.limit}&offset=${acc.offset}&all=${acc.all ? 1 : 0}`);
    $('#ainfo').textContent = `${fmtN(d.total)} accounts · showing ${acc.offset + 1}–${Math.min(acc.offset + acc.limit, d.total)}`;
    const th = (k, l) => `<th data-sort="${k}">${l}${acc.sort === k ? (acc.desc ? ' ▼' : ' ▲') : ''}</th>`;
    $('#atbl').innerHTML = `<table class="tbl"><tr>${th('handle', 'account')}${th('trust', 'trust')}<th>seed</th>${th('depth', 'depth')}<th>edges</th>${th('followers', 'followers')}<th>following</th>${th('posts', 'posts')}${th('created', 'created')}<th>labels</th></tr>
      ${d.rows.map((r) => `<tr class="clickable" data-did="${esc(r.did)}"><td><b>${esc(r.display_name || '')}</b> <span class="muted">@${esc(r.handle || r.did)}</span>${r.crawled ? '' : ' <span class="pill grey">leaf</span>'}</td><td class="num">${r.trust == null ? '–' : `<span class="pill ${band(r.trust)}">${Math.round(r.trust * 100)}</span>`}</td><td>${r.seed == null ? '' : `<span class="pill ${r.seed ? 'red' : 'green'}">${r.seed ? 'untrusted' : 'trusted'}</span>`}</td><td class="num">${r.depth == null ? '?' : r.depth}</td><td class="num">${fmtN(r.deg)}</td><td class="num">${fmtN(r.followers_count)}</td><td class="num">${fmtN(r.follows_count)}</td><td class="num">${fmtN(r.posts_count)}</td><td>${r.created_at ? r.created_at.slice(0, 10) : ''}</td><td class="small muted">${esc((JSON.parse(r.labels || '[]')).join(', '))}</td></tr>`).join('')}</table>
      <div class="row" style="margin-top:10px"><button class="btn small" id="prev" ${acc.offset === 0 ? 'disabled' : ''}>← prev</button><button class="btn small" id="next" ${acc.offset + acc.limit >= d.total ? 'disabled' : ''}>next →</button></div>`;
    $('#atbl').querySelectorAll('th[data-sort]').forEach((t) => t.addEventListener('click', () => { if (acc.sort === t.dataset.sort) acc.desc = !acc.desc; else { acc.sort = t.dataset.sort; acc.desc = true; } acc.offset = 0; loadAccounts(); }));
    $('#atbl').querySelectorAll('tr.clickable').forEach((tr) => tr.addEventListener('click', () => openNode(tr.dataset.did)));
    $('#prev').addEventListener('click', () => { acc.offset = Math.max(0, acc.offset - acc.limit); loadAccounts(); });
    $('#next').addEventListener('click', () => { acc.offset += acc.limit; loadAccounts(); });
  }

  // ---------------- seeds ----------------
  async function renderSeeds() {
    const [s, st] = await Promise.all([api('/bsky/seeds'), api('/bsky/settings')]);
    content.innerHTML = `<div class="grid cols-2"><div class="card"><h2>Seeds drive the prior q</h2><p class="small muted">+½ for untrusted, −½ for trusted, 0 otherwise. Random seeds are the placeholder you asked for; edit any of them here, in the Network drawer or in the popover on bsky.app.</p>
        <div class="row"><label class="small muted">random trusted seeds <input type="number" id="k" value="${st.scoring.random_seed_count}" min="1" max="500" style="width:70px"></label><button class="btn" id="draw">Draw new</button>
        <input id="sh" placeholder="handle" style="width:200px"><button class="btn small green" data-add="0">add trusted</button><button class="btn small red" data-add="1">add untrusted</button><button class="btn small" id="clear">clear all</button></div>
        <div class="row" style="margin-top:10px"><button class="btn" id="import">Import news outlets + verified accounts as trusted</button><span class="small muted" id="importmsg"></span></div>
        <h3 style="margin-top:12px">Automatic rules (re-derived before every run; manual and list seeds win)</h3>
        <label class="small muted" style="display:block"><input type="checkbox" id="rverified" ${st.scoring.rule_verified_trusted ? 'checked' : ''}> verified account ⇒ trusted seed</label>
        <label class="small muted" style="display:block"><input type="checkbox" id="rbots" ${st.scoring.rule_bots ? 'checked' : ''}> untrusted seed if created less than <input type="number" id="rdays" value="${st.scoring.rule_new_account_days}" min="0" step="1" style="width:60px"> days ago or fewer than <input type="number" id="rfollowers" value="${st.scoring.rule_min_followers}" min="0" style="width:70px"> followers</label>
        <label class="small muted" style="display:block"><input type="checkbox" id="auto" ${st.scoring.auto_label_seeds ? 'checked' : ''}> Bluesky moderation label (spam, bot, impersonation, …) ⇒ untrusted seed</label>
        <div class="row small muted" style="margin-top:8px">${(await api('/bsky/analytics')).seeds_by.map((x) => `<span class="pill ${x.label ? 'red' : 'green'}">${x.source} · ${x.n}</span>`).join(' ')}</div></div>
      <div class="card"><h2>Propagation parameters</h2><div class="row"><label class="small muted">α <input type="number" id="alpha" value="${st.scoring.alpha}" step="0.05" min="0" max="0.99" style="width:70px"></label>
        <label class="small muted">seed residual <input type="number" id="sres" value="${st.scoring.seed_residual}" step="0.05" min="0" max="0.5" style="width:70px"></label>
        <label class="small muted">hub damping <input type="number" id="hub" value="${st.scoring.hub_damping}" step="0.1" min="0" max="1" style="width:70px"></label><button class="btn" id="save">Save & recompute</button></div>
        <h3 style="margin-top:12px">Channel weights</h3><div class="row">${Object.entries(st.scoring.channel_weights).map(([k, v]) => `<label class="small muted">${k} <input type="number" data-w="${k}" value="${v}" step="0.05" min="0" max="1" style="width:60px"></label>`).join('')}</div></div></div>
      <div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">${s.seeds.length} seeds</h2><label class="small muted">filter <select id="sfilter"><option value="">all sources</option>${[...new Set(s.seeds.map((x) => x.source))].map((x) => `<option>${x}</option>`).join('')}</select></label></div><table class="tbl" id="stbl"><tr><th>account</th><th>label</th><th>source</th><th>trust</th><th></th></tr>${s.seeds.map((x) => `<tr><td class="clickable" data-did="${esc(x.did)}">@${esc(x.handle || x.did)} <span class="muted">${esc(x.display_name || '')}</span></td><td><span class="pill ${x.label ? 'red' : 'green'}">${x.label ? 'untrusted' : 'trusted'}</span></td><td class="muted">${esc(x.source)}</td><td class="num">${x.trust == null ? '–' : Math.round(x.trust * 100)}</td><td><button class="btn small" data-flip="${esc(x.did)}" data-label="${1 - x.label}">flip</button> <button class="btn small" data-rm="${esc(x.did)}">remove</button></td></tr>`).join('')}</table></div>`;
    const act = async (body) => { try { await api('/bsky/seeds', 'POST', body); renderSeeds(); } catch (e) { alert(e.message); } };
    $('#sfilter').addEventListener('change', () => { const v = $('#sfilter').value; $('#stbl').querySelectorAll('tr').forEach((tr, i) => { if (i === 0) return; tr.style.display = !v || tr.children[2].textContent === v ? '' : 'none'; }); });
    $('#import').addEventListener('click', async () => { $('#import').disabled = true; $('#importmsg').textContent = 'resolving outlets, searching verified accounts, back-filling profiles…'; try { const r = await api('/bsky/seeds', 'POST', { action: 'import_list' }); $('#importmsg').textContent = `${r.outlets_resolved} outlets + ${r.verified_from_search} verified accounts imported (${r.outlets_missing.length} handles not on Bluesky)`; setTimeout(renderSeeds, 1500); } catch (e) { $('#importmsg').textContent = e.message; } $('#import').disabled = false; });
    $('#draw').addEventListener('click', () => act({ action: 'random', k: Number($('#k').value) || 10 }));
    content.querySelectorAll('[data-add]').forEach((b) => b.addEventListener('click', () => { const h = $('#sh').value.trim().replace(/^@/, ''); if (h) act({ action: 'set', ident: h, label: Number(b.dataset.add) }); }));
    $('#clear').addEventListener('click', () => { if (confirm('Remove every seed?')) act({ action: 'clear' }); });
    content.querySelectorAll('[data-flip]').forEach((b) => b.addEventListener('click', () => act({ action: 'set', ident: b.dataset.flip, label: Number(b.dataset.label) })));
    content.querySelectorAll('[data-rm]').forEach((b) => b.addEventListener('click', () => act({ action: 'remove', ident: b.dataset.rm })));
    content.querySelectorAll('td.clickable').forEach((td) => td.addEventListener('click', () => openNode(td.dataset.did)));
    $('#save').addEventListener('click', async () => {
      const channel_weights = {}; content.querySelectorAll('[data-w]').forEach((i) => { channel_weights[i.dataset.w] = Number(i.value); });
      try { await api('/bsky/settings', 'POST', { scoring: { alpha: Number($('#alpha').value), seed_residual: Number($('#sres').value), hub_damping: Number($('#hub').value), auto_label_seeds: $('#auto').checked, random_seed_count: Number($('#k').value), channel_weights,
        rule_verified_trusted: $('#rverified').checked, rule_bots: $('#rbots').checked, rule_new_account_days: Number($('#rdays').value), rule_min_followers: Number($('#rfollowers').value) } }); renderSeeds(); } catch (e) { alert(e.message); }
    });
  }

  // ---------------- method ----------------
  function renderMethod() {
    content.innerHTML = `<div class="md card">
      <h2>How the Bluesky score is computed</h2>
      <p>The add-on sends the accounts on the page to this server. The server crawls the <b>public</b> Bluesky AppView (no login) <b>breadth-first</b>: for each account it reads the profile, up to <i>list pages</i>×100 followers and follows, and the last posts (replies, reposts, quotes, mentions), stores everything in SQLite, then enqueues the closest neighbours (mutual follows first) one level deeper, with lower priority than anything the page asked for. A crawled account is not fetched again before its TTL.</p>
      <p>On the stored graph the step-1 formula runs unchanged:</p>
      <p><code>r = (1 − α) q + α P r</code> &nbsp;·&nbsp; <code>trust = ½ − r ∈ [0, 1]</code></p>
      <ul><li><code>q</code>: prior residual. Seeds: +½ untrusted, −½ trusted. Trusted by default: the curated list of news outlets, verified accounts found by search, and every account the AppView reports as verified (rule). Untrusted by default: accounts created less than 7 days ago, accounts with fewer than 20 followers (when the count is known), and accounts carrying a moderation label. Manual clicks always win. Every other account: <code>λ (p_local − ½)</code> with λ = 0.5 and <code>p_local = ½ + ½ tanh(Σ evidence)</code>, a hand-set profile prior (created &lt; 30 d +0.6, &lt; 180 d +0.2, &gt; 2 y −0.3; default handle ending in digits +0.3, custom domain −0.3; no avatar +0.3, no display name +0.15; follows &gt; 5× followers +0.4, large audience −0.2, no posts +0.2, &gt; 50 posts/day +0.4, &gt; 20 +0.2; moderation label +1.0).</li>
      <li><code>P</code>: row-normalised gather matrix over typed channels (mutual, followed-by, follows, replies, reposts, quotes, mentions in both directions), each edge scaled by the channel's homophily weight, each sender damped by 1/deg<sup>½</sup>.</li>
      <li>Jacobi iteration converges at rate α; the fixed point is unique and |r| ≤ ½.</li></ul>
      <p>The linearity gives the exact decomposition shown in the drawer: <code>r_v = (1−α) q_v + α Σ_c w_c Σ_u G_c[v,u]/D_v · r_u</code>, one row per channel, one line per neighbour.</p>
      <h3>Why scores stay compressed around 50</h3>
      <p>A node keeps only (1−α) of its own evidence: a seed alone scores 60 or 40 with α = 0.8, a profile prior at most ±5 points. Its neighbours receive that residual divided by their degree and damped by the sender's degree, and leaves (seen in one list only) echo a single neighbour. Accounts with fewer than three stored edges are reported as "not enough graph" (grey "?" badge) until their crawl runs. Use the contrast slider in the Network tab to see small differences, add seeds, or lower α.</p>
      <h3>Approximations</h3>
      <table><tr><th>item</th><th>value</th></tr>
      <tr><td>followers / follows read per account</td><td>list pages × 100 (flag "lists truncated")</td></tr><tr><td>posts read per account</td><td>feed limit (default 100); likes not crawled</td></tr>
      <tr><td>prior of non-seed accounts</td><td>hand-set profile evidence (weights above), λ = 0.5; leaves only have handle, name, avatar, creation date</td></tr>
      <tr><td>"not enough graph"</td><td>uncrawled accounts with &lt; 3 stored edges</td></tr><tr><td>BFS expansion</td><td>25 neighbours per page account, 6 per deeper node, up to max depth (settings)</td></tr><tr><td>channel weights</td><td>hand-set step-1 prior, not re-estimated on Bluesky</td></tr>
      <tr><td>seed rules</td><td>verified ⇒ trusted; age &lt; 7 d or followers &lt; 20 ⇒ untrusted (thresholds in Seeds tab; follower rule only when the count is known, i.e. crawled/resolved accounts; ISO date comparison assumes UTC "Z" timestamps)</td></tr>
      <tr><td>curated list</td><td>~230 outlet domain handles (trusted_lists.py) validated live + verified accounts matching 25 news-related search terms</td></tr><tr><td>rate</td><td>4 req/s (public limit ≈ 10 req/s per IP)</td></tr></table>
      <p class="small muted">Code: backend/trustscore/propagation.py (formula), backend/trustscore/bluesky/ (crawler, store, scoring, analytics). Data: data/bluesky/bluesky.sqlite.</p></div>`;
  }

  let resizeT = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => {
      if (state.tab !== 'network' || !net) return;
      const box = net.canvas.parentElement, W = box.clientWidth, H = box.clientHeight, dpr = window.devicePixelRatio || 1;
      if (W === net.W && H === net.H) return;
      net.canvas.width = W * dpr; net.canvas.height = H * dpr; net.W = W; net.H = H; net.dpr = dpr; draw();
    }, 150);
  });
  setTab(state.tab);
})();
