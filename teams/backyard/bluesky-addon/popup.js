const $ = (s) => document.querySelector(s);
const api = (path, method, body) => new Promise((res, rej) => chrome.runtime.sendMessage({ type: 'api', path, method, body }, (r) => {
  if (!r || !r.ok) rej(new Error((r && (r.error || 'HTTP ' + r.status)) || 'no response')); else res(r.data);
}));
const msg = (t) => { $('#msg').textContent = t || ''; };
const CRAWL_KEYS = ['max_depth', 'expand_per_node', 'list_pages', 'feed_limit', 'ttl_hours', 'requests_per_second'];

async function refresh() {
  try {
    const st = await api('/bsky/status');
    const s = st.store, c = st.crawler, r = st.last_run;
    $('#status').innerHTML = `<span>graph</span><b>${s.actors} accounts · ${s.follows} follows · ${s.interactions} interactions</b>` +
      `<span>crawled</span><b>${s.crawled} accounts · queue ${c.queue.total}${c.current ? ' · working' : ''}</b>` +
      `<span>API calls</span><b>${c.api_calls}${c.last_error ? ' · last error: ' + c.last_error.slice(0, 60) : ''}</b>` +
      `<span>last run</span><b>${r ? new Date(r.computed_at * 1000).toLocaleTimeString() + ' · ' + r.n_nodes + ' nodes · ' + r.n_seeds + ' seeds · ' + r.iterations + ' it.' : 'none yet'}</b>`;
    CRAWL_KEYS.forEach((k) => { $('#' + k).value = c.settings[k]; });
    $('#alpha').value = st.scoring.alpha;
    $('#k').value = st.scoring.random_seed_count;
    $('#autoLabel').checked = !!st.scoring.auto_label_seeds;
    $('#ruleVerified').checked = !!st.scoring.rule_verified_trusted; $('#ruleBots').checked = !!st.scoring.rule_bots;
    $('#ruleDays').value = st.scoring.rule_new_account_days; $('#ruleFollowers').value = st.scoring.rule_min_followers;
    const seeds = (await api('/bsky/seeds')).seeds;
    $('#seeds').innerHTML = seeds.map((x) => `<li><span class="tag t${x.label}">${x.label ? 'untrusted' : 'trusted'}</span>` +
      `<span class="h" title="${x.did}">@${x.handle || x.did}</span><span class="src">${x.source}${x.trust != null ? ' · ' + Math.round(x.trust * 100) : ''}</span>` +
      `<button data-flip="${x.did}" data-label="${1 - x.label}">flip</button><button data-rm="${x.did}">×</button></li>`).join('') || '<li class="src">no seeds yet — draw random ones or add handles</li>';
    $('#seeds').querySelectorAll('[data-rm]').forEach((b) => b.addEventListener('click', () => act({ action: 'remove', ident: b.dataset.rm })));
    $('#seeds').querySelectorAll('[data-flip]').forEach((b) => b.addEventListener('click', () => act({ action: 'set', ident: b.dataset.flip, label: Number(b.dataset.label) })));
    msg('');
  } catch (e) { msg('Server unreachable: ' + e.message); }
}
async function act(body) {
  try { await api('/bsky/seeds', 'POST', body); await chrome.storage.local.set({ seedsVersion: Date.now() }); await refresh(); }
  catch (e) { msg(e.message); }
}

(async () => {
  const v = await chrome.storage.local.get(['server', 'enabled']);
  $('#server').value = v.server || 'http://127.0.0.1:8010';
  $('#enabled').checked = v.enabled !== false;
  $('#enabled').addEventListener('change', () => chrome.storage.local.set({ enabled: $('#enabled').checked }));
  $('#saveServer').addEventListener('click', async () => { await chrome.storage.local.set({ server: $('#server').value.trim() }); refresh(); });
  const dashUrl = async (q) => ((await chrome.storage.local.get('server')).server || 'http://127.0.0.1:8010').replace(/\/$/, '') + '/addon/dashboard.html' + (q || '');
  $('#openDash').addEventListener('click', async () => chrome.tabs.create({ url: await dashUrl('?tab=overview') }));
  $('#openPage').addEventListener('click', async () => {
    let idents = [];
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.id) idents = (await chrome.tabs.sendMessage(tab.id, { type: 'visible' })) || [];
    } catch (e) { /* not a bsky.app tab */ }
    if (!idents.length) { msg('Open a bsky.app page with scored accounts first.'); return; }
    chrome.tabs.create({ url: await dashUrl('?tab=network&hops=1&focus=' + encodeURIComponent(idents.slice(0, 12).join(','))) });
  });
  $('#draw').addEventListener('click', () => act({ action: 'random', k: Number($('#k').value) || 10 }));
  document.querySelectorAll('.addSeed').forEach((b) => b.addEventListener('click', () => {
    const h = $('#seedHandle').value.trim().replace(/^@/, ''); if (h) act({ action: 'set', ident: h, label: Number(b.dataset.label) });
  }));
  $('#clearSeeds').addEventListener('click', () => act({ action: 'clear' }));
  $('#importList').addEventListener('click', async () => {
    $('#importList').disabled = true; msg('Importing (resolving handles, searching verified accounts)…');
    try { const r = await api('/bsky/seeds', 'POST', { action: 'import_list' }); msg(`Imported ${r.outlets_resolved} outlets + ${r.verified_from_search} verified accounts (${r.outlets_missing.length} handles not on Bluesky).`); await chrome.storage.local.set({ seedsVersion: Date.now() }); refresh(); }
    catch (e) { msg(e.message); } finally { $('#importList').disabled = false; }
  });
  $('#recompute').addEventListener('click', async () => { try { await api('/bsky/recompute', 'POST'); await chrome.storage.local.set({ seedsVersion: Date.now() }); refresh(); } catch (e) { msg(e.message); } });
  $('#saveSettings').addEventListener('click', async () => {
    const crawler = {}; CRAWL_KEYS.forEach((k) => { crawler[k] = Number($('#' + k).value); });
    try {
      await api('/bsky/settings', 'POST', { crawler, scoring: { alpha: Number($('#alpha').value), random_seed_count: Number($('#k').value), auto_label_seeds: $('#autoLabel').checked,
        rule_verified_trusted: $('#ruleVerified').checked, rule_bots: $('#ruleBots').checked, rule_new_account_days: Number($('#ruleDays').value), rule_min_followers: Number($('#ruleFollowers').value) } });
      await chrome.storage.local.set({ seedsVersion: Date.now() }); refresh();
    } catch (e) { msg(e.message); }
  });
  refresh();
  setInterval(refresh, 5000);
})();
