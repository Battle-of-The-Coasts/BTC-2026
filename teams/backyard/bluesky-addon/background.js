// Service worker: relays API calls from the content script / popup to the local scoring server
// (keeps the page's CSP out of the way and centralises the server URL).
const DEFAULT_SERVER = 'http://127.0.0.1:8010';

async function serverUrl() {
  const { server } = await chrome.storage.local.get('server');
  return (server || DEFAULT_SERVER).replace(/\/$/, '');
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === 'api') {
    (async () => {
      try {
        const base = await serverUrl();
        const r = await fetch(base + msg.path, {
          method: msg.method || 'GET',
          headers: { 'Content-Type': 'application/json' },
          body: msg.body ? JSON.stringify(msg.body) : undefined,
        });
        const data = await r.json().catch(() => null);
        sendResponse({ ok: r.ok, status: r.status, data });
      } catch (e) {
        sendResponse({ ok: false, status: 0, error: String(e) });
      }
    })();
    return true;   // async response
  }
});
