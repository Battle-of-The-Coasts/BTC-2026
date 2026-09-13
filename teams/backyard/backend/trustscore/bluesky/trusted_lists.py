"""Curated trusted seeds: news outlets (domain handles on Bluesky) plus verified accounts found by actor search.

`import_trusted` resolves every handle through the AppView (unknown ones are skipped), stores the full profiles,
marks them trusted seeds (source 'list' / 'list:search', never overriding a manual seed) and queues them for a crawl
so their neighbourhoods join the graph. It also back-fills the `verified` flag of already crawled accounts.
"""
from __future__ import annotations

import logging

from .client import BskyClient, BskyError
from .store import Store

log = logging.getLogger("bluesky.lists")

OUTLETS = [
    # US national
    "nytimes.com", "washingtonpost.com", "wsj.com", "usatoday.com", "latimes.com", "chicagotribune.com", "bostonglobe.com",
    "seattletimes.com", "sfchronicle.com", "startribune.com", "dallasnews.com", "houstonchronicle.com", "miamiherald.com",
    "ajc.com", "denverpost.com", "tampabay.com", "inquirer.com", "baltimoresun.com", "azcentral.com", "oregonlive.com",
    # wires, broadcast, digital
    "apnews.com", "reuters.com", "afp.com", "npr.org", "pbs.org", "cbsnews.com", "nbcnews.com", "abcnews.com", "cnn.com",
    "msnbc.com", "cnbc.com", "bloomberg.com", "marketwatch.com", "axios.com", "politico.com", "thehill.com", "propublica.org",
    "theatlantic.com", "newyorker.com", "wired.com", "theverge.com", "techcrunch.com", "arstechnica.com", "engadget.com",
    "404media.co", "semafor.com", "vox.com", "slate.com", "salon.com", "motherjones.com", "thenation.com", "theintercept.com",
    "rollingstone.com", "variety.com", "hollywoodreporter.com", "time.com", "forbes.com", "fortune.com", "businessinsider.com",
    "economist.com", "ft.com", "newsweek.com", "huffpost.com", "thedailybeast.com", "nymag.com", "thecut.com", "vulture.com",
    "theathletic.com", "espn.com", "si.com", "scientificamerican.com", "nature.com", "science.org", "newscientist.com",
    "quantamagazine.org", "statnews.com", "themarshallproject.org", "texastribune.org", "calmatters.org", "thecity.nyc",
    "gothamist.com", "chalkbeat.org", "the19th.org", "grist.org", "insideclimatenews.org", "bellingcat.com", "niemanlab.org",
    "cjr.org", "poynter.org", "factcheck.org", "snopes.com", "politifact.com", "theconversation.com", "thebulwark.com",
    "reason.com", "nationalreview.com", "thedispatch.com", "puck.news", "punchbowl.news", "notus.org", "pressgazette.co.uk",
    # UK & Ireland
    "theguardian.com", "bbc.com", "bbc.co.uk", "independent.co.uk", "telegraph.co.uk", "thetimes.com", "thetimes.co.uk",
    "channel4.com", "itv.com", "news.sky.com", "newstatesman.com", "spectator.co.uk", "private-eye.co.uk", "irishtimes.com",
    "rte.ie", "thejournal.ie", "theferret.scot", "bylinetimes.com", "opendemocracy.net", "tortoisemedia.com",
    # Europe
    "lemonde.fr", "liberation.fr", "lefigaro.fr", "mediapart.fr", "francetvinfo.fr", "france24.com", "rfi.fr", "lesechos.fr",
    "courrierinternational.com", "nouvelobs.com", "lexpress.fr", "humanite.fr", "lacroix.com", "ouest-france.fr", "20minutes.fr",
    "spiegel.de", "zeit.de", "sueddeutsche.de", "faz.net", "taz.de", "tagesschau.de", "dw.com", "handelsblatt.com",
    "elpais.com", "elmundo.es", "lavanguardia.com", "eldiario.es", "corriere.it", "repubblica.it", "ilpost.it", "ansa.it",
    "nrc.nl", "volkskrant.nl", "nos.nl", "derstandard.at", "nzz.ch", "srf.ch", "rts.ch", "politico.eu", "euronews.com",
    "euractiv.com", "svt.se", "dr.dk", "nrk.no", "yle.fi", "publico.pt", "kathimerini.gr", "wyborcza.pl", "hvg.hu",
    # Americas, Asia-Pacific, Africa, Middle East
    "cbc.ca", "theglobeandmail.com", "thestar.com", "nationalpost.com", "ledevoir.com", "lapresse.ca", "radio-canada.ca",
    "abc.net.au", "smh.com.au", "theage.com.au", "sbs.com.au", "nzherald.co.nz", "rnz.co.nz", "stuff.co.nz",
    "aljazeera.com", "haaretz.com", "timesofisrael.com", "scmp.com", "japantimes.co.jp", "nikkei.com", "thehindu.com",
    "indianexpress.com", "scroll.in", "thewire.in", "dawn.com", "folha.uol.com.br", "estadao.com.br", "clarin.com",
    "lanacion.com.ar", "eltiempo.com", "elmercurio.com", "dailymaverick.co.za", "news24.com", "mg.co.za", "premiumtimesng.com",
]

SEARCH_TERMS = ["news", "newspaper", "journal", "magazine", "times", "tribune", "gazette", "herald", "post", "radio",
                "television", "press", "journalist", "reporter", "correspondent", "editor", "newsroom", "public media",
                "investigative", "fact check", "science news", "tech news", "sports news", "business news", "local news"]


def resolve_many(client: BskyClient, handles: list[str]) -> tuple[list[dict], list[str]]:
    """getProfiles in batches; a batch with an unresolvable handle fails, so fall back to single lookups."""
    found = []
    for i in range(0, len(handles), 25):
        batch = handles[i:i + 25]
        try:
            found.extend(client.get_profiles(batch))   # unknown handles are silently dropped by the AppView
            continue
        except BskyError:
            pass
        for h in batch:
            try:
                found.append(client.get_profile(h))
            except BskyError:
                pass
    got = {p.get("handle", "").lower() for p in found} | {p.get("did") for p in found}
    missing = [h for h in handles if h.lower() not in got]
    return found, missing


def import_trusted(store: Store, client: BskyClient, crawler=None, expand_search: bool = True, backfill: bool = True,
                   handles: list[str] | None = None) -> dict:
    handles = handles or OUTLETS
    profiles, missing = resolve_many(client, handles)
    n_list = 0
    for p in profiles:
        store.upsert_actor(p, depth=0, full_profile=True)
        store.set_seed(p["did"], 0, "list", keep_manual=True)
        n_list += 1
        if crawler is not None and not store.is_fresh(p["did"], crawler.settings.ttl_hours * 3600):
            store.enqueue(p["did"], 1, 0)
    n_search, seen = 0, set()
    if expand_search:
        for term in SEARCH_TERMS:
            try:
                res = client.get("app.bsky.actor.searchActors", q=term, limit=100)
            except BskyError as e:
                log.warning("searchActors %r failed: %s", term, e)
                continue
            for a in res.get("actors", []):
                did = a.get("did")
                if not did or did in seen:
                    continue
                seen.add(did)
                if (a.get("verification") or {}).get("verifiedStatus") != "valid":
                    continue
                store.upsert_actor(a, depth=0)
                store.set_seed(did, 0, "list:search", keep_manual=True, keep=("manual", "list"))
                n_search += 1
                if crawler is not None and not store.is_fresh(did, crawler.settings.ttl_hours * 3600):
                    store.enqueue(did, 2, 0)   # background, but before every depth >= 1 expansion
    n_backfill = 0
    if backfill:
        dids = [r[0] for r in store.q("SELECT a.did FROM actors a JOIN crawl c ON c.did=a.did WHERE c.followers_at IS NOT NULL "
                                      "AND a.verified IS NULL")]
        found, _ = resolve_many(client, dids)
        for p in found:
            store.upsert_actor(p, depth=None, full_profile=True)
            n_backfill += 1
    if crawler is not None:
        crawler._wake.set()
    return {"outlets_resolved": n_list, "outlets_missing": missing, "verified_from_search": n_search,
            "profiles_backfilled": n_backfill}
