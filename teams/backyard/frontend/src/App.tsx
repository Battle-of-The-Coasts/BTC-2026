import { useEffect, useState } from 'react'
import { api } from './api'
import type { Clusters, DatasetInfo, GraphData, Metrics, Summary } from './api'
import Overview from './pages/Overview'
import Network from './pages/Network'
import ClustersPage from './pages/Clusters'
import Accounts from './pages/Accounts'
import Method from './pages/Method'
import NodeDetails from './components/NodeDetails'

type Tab = 'overview' | 'network' | 'clusters' | 'accounts' | 'method'
export type Highlight = { kind: 'community' | 'block' | 'subset'; id: number | string } | null

const TABS: [Tab, string][] = [['overview', 'Overview'], ['network', 'Network'], ['clusters', 'Clusters'], ['accounts', 'Accounts'], ['method', 'Method']]

export default function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [ds, setDs] = useState('')
  const [tab, setTab] = useState<Tab>('overview')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [clusters, setClusters] = useState<Clusters | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [highlight, setHighlight] = useState<Highlight>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.datasets().then((d) => { setDatasets(d); if (d.length) setDs(d[0].name) }).catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (!ds) return
    setSummary(null); setMetrics(null); setGraph(null); setClusters(null); setSelected(null); setHighlight(null); setError(null)
    Promise.all([api.summary(ds), api.metrics(ds), api.graph(ds), api.clusters(ds)])
      .then(([s, m, g, c]) => { setSummary(s); setMetrics(m); setGraph(g); setClusters(c) })
      .catch((e) => setError(String(e)))
  }, [ds])

  const ready = summary && metrics && graph && clusters

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">Trust<span>Score</span> <span className="muted small">· network bot detection, step 1</span></div>
        <select value={ds} onChange={(e) => setDs(e.target.value)}>
          {datasets.map((d) => <option key={d.name} value={d.name}>{d.name}</option>)}
        </select>
        <div className="tabs">
          {TABS.map(([t, name]) => <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{name}</button>)}
        </div>
        {summary && <div className="muted small" style={{ marginLeft: 'auto' }}>
          {summary.dataset.n_nodes.toLocaleString()} nodes · {summary.dataset.n_labeled.toLocaleString()} labelled · {summary.dataset.n_bots.toLocaleString()} bots
        </div>}
      </div>
      <div className="main">
        <div className="content" style={tab === 'network' ? { padding: 0, overflow: 'hidden' } : undefined}>
          {error && <div className="error">{error}. Is the API running (python backend/scripts/serve.py) and has the pipeline produced data/processed/&lt;dataset&gt;?</div>}
          {!error && !ready && datasets.length > 0 && <div className="loading">Loading {ds}…</div>}
          {!error && datasets.length === 0 && <div className="loading">No processed dataset found. Run <code>python backend/scripts/run_pipeline.py cresci-2015</code>.</div>}
          {ready && tab === 'overview' && <Overview summary={summary} metrics={metrics} clusters={clusters} />}
          {ready && tab === 'network' && <Network ds={ds} graph={graph} summary={summary} metrics={metrics} clusters={clusters}
            highlight={highlight} setHighlight={setHighlight} selected={selected} onSelect={setSelected} />}
          {ready && tab === 'clusters' && <ClustersPage ds={ds} clusters={clusters} onSelect={setSelected}
            onHighlight={(h) => { setHighlight(h); setTab('network') }} />}
          {ready && tab === 'accounts' && <Accounts ds={ds} summary={summary} onSelect={setSelected} />}
          {tab === 'method' && <Method />}
        </div>
        {selected !== null && ds && (
          <div className="drawer">
            <NodeDetails ds={ds} idx={selected} onClose={() => setSelected(null)} onSelect={setSelected} />
          </div>
        )}
      </div>
    </div>
  )
}
