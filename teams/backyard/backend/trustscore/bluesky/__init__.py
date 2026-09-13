"""Bluesky live crawler + trust scoring behind the browser add-on.

Modules
  client   : rate-limited wrapper over the public AppView (no auth needed)
  store    : SQLite persistence (actors, follows, interactions, crawl state, BFS queue, seeds, scores)
  crawler  : breadth-first expansion of the graph around the accounts the add-on sees
  scoring  : builds typed edge channels from the store and runs the step-1 propagation formula
  api      : FastAPI app used by the add-on
"""
