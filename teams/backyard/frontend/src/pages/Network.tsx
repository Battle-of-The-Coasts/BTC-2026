import { useEffect, useMemo, useRef, useState } from 'react'
import Graph from 'graphology'
import Sigma from 'sigma'
import { api, catColor, fmt, LABEL_COLORS, LABEL_NAMES, PALETTE, trustColor } from '../api'
import type { Clusters, GraphData, Metrics, Summary, WhatIfResult } from '../api'
import type { Highlight } from '../App'

type ColorMode = 'calibrated' | 'trust' | 'label' | 'community' | 'subset'
const EDGE_COLORS: Record<string, string> = { similar: '#2c3650', mutual: '#2b4a3f', follow: '#31333f', follow_hub: '#40362c' }
const EDGE_NAMES: Record<string, string> = { similar: 'co-following similarity', mutual: 'mutual follow', follow: 'one-way follow', follow_hub: 'follows external hub' }
const DEFAULT_EDGES = new Set(['similar', 'mutual'])
const DIM = '#1c1f2a'
const MAX_NODE_SIZE = 4.5

// node radius in px at camera ratio 1: log of the full-graph degree, capped so that 10k nodes never fuse into a blob
const nodeSize = (deg: number) => Math.min(MAX_NODE_SIZE, 0.6 + 0.45 * Math.log1p(Math.max(deg, 0)))
const edgeSize = (ty: string, w: number) => (ty === 'similar' ? 0.3 + 0.4 * w : ty === 'mutual' ? 0.3 : 0.25)

// robust bounding box (1st-99th percentile, padded) so that a few far-away nodes cannot shrink the whole picture
function robustBBox(nodes: GraphData['nodes']) {
  const pct = (vals: number[], p: number) => { const s = [...vals].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))] }
  const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y)
  const x0 = pct(xs, 0.01), x1 = pct(xs, 0.99), y0 = pct(ys, 0.01), y1 = pct(ys, 0.99)
  const half = Math.max(x1 - x0, y1 - y0, 1e-6) * 0.64, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2
  return { x: [cx - half, cx + half] as [number, number], y: [cy - half, cy + half] as [number, number] }
}

type Props = {
  ds: string; graph: GraphData; summary: Summary; metrics: Metrics; clusters: Clusters
  highlight: Highlight; setHighlight: (h: Highlight) => void; selected: number | null; onSelect: (i: number | null) => void
}

export default function Network({ ds, graph, summary, metrics, clusters, highlight, setHighlight, selected, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)
  const graphRef = useRef<Graph | null>(null)
  const [mode, setMode] = useState<ColorMode>('calibrated')
  const [edgeOn, setEdgeOn] = useState<Record<string, boolean>>(() => Object.fromEntries(graph.edge_types.map((t) => [t, DEFAULT_EDGES.has(t)])))
  const [search, setSearch] = useState('')
  const [members, setMembers] = useState<Set<number> | null>(null)
  const [whatIf, setWhatIf] = useState<WhatIfResult | null>(null)
  const [showWhatIf, setShowWhatIf] = useState(false)
  const [busy, setBusy] = useState(false)
  const pc = summary.config.propagation
  const channels = useMemo(() => Object.keys(metrics.channel_weights['0'] ?? pc.relation_weights), [metrics, pc])
  const sel = metrics.selected_params?.['0']
  const [alpha, setAlpha] = useState(sel?.alpha ?? pc.alpha)
  const [lam, setLam] = useState(sel?.lam ?? pc.lam)
  const [weights, setWeights] = useState<Record<string, number>>(() => ({ ...(metrics.channel_weights['0'] ?? pc.relation_weights) }))
  const subsetIndex = useMemo(() => { const s = Array.from(new Set(graph.nodes.map((n) => n.s))).sort(); return Object.fromEntries(s.map((v, i) => [v, i])) }, [graph])
  const nEdgesShown = useMemo(() => graph.edges.filter((e) => edgeOn[e[2]]).length, [graph, edgeOn])

  // build graph + renderer once per dataset
  useEffect(() => {
    if (!container.current) return
    const g = new Graph({ multi: false, type: 'undirected' })
    for (const n of graph.nodes) {
      const p = n.p ?? (1 - n.t)
      g.addNode(String(n.i), { x: n.x, y: n.y, size: nodeSize(n.d), label: n.n ? `@${n.n}` : n.id, t: n.t, p, l: n.l, c: n.c, s: n.s, b: n.b, color: trustColor(1 - p) })
    }
    for (const [s, t, ty, w] of graph.edges) {
      const a = String(s), b = String(t)
      if (a === b || !g.hasNode(a) || !g.hasNode(b) || g.hasEdge(a, b)) continue
      g.addEdge(a, b, { ty, w, size: edgeSize(ty, w), color: EDGE_COLORS[ty] ?? '#333' })
    }
    graphRef.current = g
    const renderer = new Sigma(g, container.current, {
      renderEdgeLabels: false, labelRenderedSizeThreshold: 5, labelDensity: 0.25, labelGridCellSize: 110, labelColor: { color: '#c9ccd8' },
      defaultEdgeColor: '#2a2f3d', minCameraRatio: 0.01, maxCameraRatio: 2.5, allowInvalidContainer: true, zIndex: true,
      hideEdgesOnMove: true, labelFont: 'Inter, system-ui, sans-serif', labelSize: 11,
    })
    renderer.setCustomBBox(robustBBox(graph.nodes))
    renderer.on('clickNode', ({ node }) => onSelect(Number(node)))
    renderer.on('clickStage', () => onSelect(null))
    sigmaRef.current = renderer
    return () => { renderer.kill(); sigmaRef.current = null; graphRef.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph])

  // highlight members (community / block / subset)
  useEffect(() => {
    if (!highlight) { setMembers(null); return }
    if (highlight.kind === 'subset') { setMembers(new Set(graph.nodes.filter((n) => n.s === highlight.id).map((n) => n.i))); return }
    api.clusterMembers(ds, Number(highlight.id), highlight.kind).then((m) => setMembers(new Set(m.members))).catch(() => setMembers(null))
  }, [highlight, ds, graph])

  // reducers
  useEffect(() => {
    const r = sigmaRef.current
    if (!r) return
    const sel = selected === null ? null : String(selected)
    const neighbours = sel && r.getGraph().hasNode(sel) ? new Set(r.getGraph().neighbors(sel)) : null
    r.setSetting('nodeReducer', (node, data) => {
      const d = { ...data }
      const idx = Number(node)
      let t = data.t as number
      if (whatIf && whatIf.trust[node] !== undefined) t = whatIf.trust[node]
      const cal = (whatIf && whatIf.p_bot && whatIf.p_bot[node] !== undefined) ? 1 - whatIf.p_bot[node] : 1 - (data.p as number)
      d.color = mode === 'calibrated' ? trustColor(cal) : mode === 'trust' ? trustColor(t) : mode === 'label' ? LABEL_COLORS[data.l as number] : mode === 'community' ? catColor(data.c as number) : PALETTE[(subsetIndex[data.s as string] ?? 0) % PALETTE.length]
      if (members && !members.has(idx)) { d.color = DIM; d.label = ''; d.zIndex = 0 } else if (members) { d.zIndex = 2 }
      if (neighbours && neighbours.has(node)) { d.zIndex = 3; d.forceLabel = true }
      if (sel === node) { d.highlighted = true; d.size = Math.max((data.size as number) * 2.2, 8); d.zIndex = 4; d.forceLabel = true }
      return d
    })
    r.setSetting('edgeReducer', (edge, data) => {
      const d = { ...data }
      if (!edgeOn[data.ty as string]) d.hidden = true
      const [a, b] = r.getGraph().extremities(edge)
      if (members && !members.has(Number(a)) && !members.has(Number(b))) d.hidden = true
      if (sel && (a === sel || b === sel)) { d.hidden = false; d.color = '#8fa3ff'; d.size = Math.max((data.size as number) * 2, 1.2); d.zIndex = 1 }
      return d
    })
    r.refresh()
  }, [mode, edgeOn, members, selected, whatIf, subsetIndex])

  // focus on selection coming from other tabs
  useEffect(() => {
    const r = sigmaRef.current
    if (!r || selected === null || !r.getGraph().hasNode(String(selected))) return
    const p = r.getNodeDisplayData(String(selected))
    if (p) r.getCamera().animate({ x: p.x, y: p.y, ratio: Math.min(r.getCamera().ratio, 0.25) }, { duration: 400 })
  }, [selected])

  const doSearch = () => {
    const g = graphRef.current
    if (!g || !search) return
    const q = search.toLowerCase().replace(/^@/, '')
    const hit = g.findNode((n, a) => String(a.label).toLowerCase().replace(/^@/, '') === q || n === q) ?? g.findNode((_n, a) => String(a.label).toLowerCase().includes(q))
    if (hit) onSelect(Number(hit))
  }
  const runWhatIf = async () => {
    setBusy(true)
    try { setWhatIf(await api.whatif(ds, { alpha, lam, weights, fold: 0 })) } finally { setBusy(false) }
  }
  const resetWhatIf = () => { setWhatIf(null); setAlpha(sel?.alpha ?? pc.alpha); setLam(sel?.lam ?? pc.lam); setWeights({ ...(metrics.channel_weights['0'] ?? pc.relation_weights) }) }
  const cam = () => sigmaRef.current?.getCamera()
  const communityOptions = clusters.communities.slice(0, 40)
  const base = metrics.crossfit.trust_score

  return (
    <div className="netwrap">
      <div className="toolbar">
        <span className="small muted">colour</span>
        <select value={mode} onChange={(e) => setMode(e.target.value as ColorMode)}>
          <option value="calibrated">calibrated trust (1 − P(bot))</option><option value="trust">raw formula trust (½ − r)</option><option value="label">ground-truth label</option><option value="community">community</option><option value="subset">collection subset</option>
        </select>
        <span className="small muted">edges</span>
        {graph.edge_types.map((t) => <label key={t} className="small"><input type="checkbox" checked={!!edgeOn[t]} onChange={(e) => setEdgeOn({ ...edgeOn, [t]: e.target.checked })} /> <span style={{ color: EDGE_COLORS[t] }}>■</span> {EDGE_NAMES[t] ?? t}</label>)}
        <span className="small muted">focus</span>
        <select value={highlight ? `${highlight.kind}:${highlight.id}` : ''} onChange={(e) => { const v = e.target.value; if (!v) setHighlight(null); else { const [k, id] = v.split(':'); setHighlight({ kind: k as 'community' | 'block' | 'subset', id: k === 'subset' ? id : Number(id) }) } }}>
          <option value="">everything</option>
          {Object.keys(subsetIndex).filter((s) => s).map((s) => <option key={s} value={`subset:${s}`}>subset {s}</option>)}
          {communityOptions.map((c) => <option key={c.id} value={`community:${c.id}`}>community {c.id} (n={c.size}, bot {Math.round(100 * c.predicted_bot_share)}%)</option>)}
          {clusters.dense_blocks.map((b) => <option key={b.id} value={`block:${b.id}`}>dense block {b.id} (n={b.size})</option>)}
        </select>
        <input placeholder="find @handle or id" value={search} onChange={(e) => setSearch(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && doSearch()} style={{ width: 160 }} />
        <button className="btn small" onClick={doSearch}>go</button>
        <button className={`btn small${whatIf ? ' active' : ''}`} onClick={() => setShowWhatIf(!showWhatIf)} title="re-propagate with other parameters and recolour the nodes">
          {showWhatIf ? '▾' : '▸'} what-if{whatIf ? ' (active)' : ''}
        </button>
        <span style={{ marginLeft: 'auto' }} className="row">
          <button className="btn small" onClick={() => cam()?.animatedZoom({ duration: 250 })}>+</button>
          <button className="btn small" onClick={() => cam()?.animatedUnzoom({ duration: 250 })}>−</button>
          <button className="btn small" onClick={() => cam()?.animatedReset({ duration: 250 })}>fit</button>
        </span>
      </div>
      {showWhatIf && <div className="toolbar" style={{ borderTop: '1px solid var(--border)', background: 'var(--bg2)' }}>
        <span className="small" style={{ fontWeight: 600 }} title="Exploratory only: re-propagates with fold-0 seeds and reports metrics on fold-0 held-out labels; the pipeline never uses this to choose parameters. F1 uses the pipeline's fold-0 calibration.">What-if (exploratory, fold 0)</span>
        <label className="slider">α propagation {alpha.toFixed(2)}<input type="range" min={0} max={0.95} step={0.05} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} /></label>
        <label className="slider">λ local prior {lam.toFixed(2)}<input type="range" min={0} max={1} step={0.05} value={lam} onChange={(e) => setLam(Number(e.target.value))} /></label>
        {channels.map((c) => <label key={c} className="slider">{c} {(weights[c] ?? 0).toFixed(2)}<input type="range" min={0} max={1} step={0.05} value={weights[c] ?? 0} onChange={(e) => setWeights({ ...weights, [c]: Number(e.target.value) })} /></label>)}
        <button className="btn small" disabled={busy} onClick={runWhatIf}>{busy ? 'computing…' : 'recompute'}</button>
        <button className="btn small" onClick={resetWhatIf}>reset</button>
        <span className="small muted">
          {whatIf ? <>held-out fold 0: AUC <b>{fmt(whatIf.metrics.auc, 4)}</b> · AP {fmt(whatIf.metrics.ap, 4)} · F1 {fmt(whatIf.metrics.f1, 3)} · {whatIf.iterations} iterations (colours updated)</> : <>current pipeline (all folds): AUC {fmt(base.auc, 4)} · AP {fmt(base.ap, 4)}</>}
        </span>
      </div>}
      <div className="netcanvas">
        <div className="sigma" ref={container} />
        <div style={{ position: 'absolute', left: 10, bottom: 10, right: 10 }} className="legend">
          {(mode === 'trust' || mode === 'calibrated') && <><span><span className="sw" style={{ background: trustColor(0) }} />0 bot-like</span><span><span className="sw" style={{ background: trustColor(0.5) }} />0.5 neutral</span><span><span className="sw" style={{ background: trustColor(1) }} />1 trusted</span></>}
          {mode === 'label' && Object.entries(LABEL_NAMES).map(([k, v]) => <span key={k}><span className="sw" style={{ background: LABEL_COLORS[Number(k)] }} />{v}</span>)}
          {mode === 'subset' && Object.entries(subsetIndex).map(([s, i]) => <span key={s}><span className="sw" style={{ background: PALETTE[i % PALETTE.length] }} />{s || 'external hub'}</span>)}
          {mode === 'community' && <span>colour = Leiden community (external hubs grey)</span>}
          <span className="muted">· node size = log degree · {graph.nodes.length.toLocaleString()} nodes, {nEdgesShown.toLocaleString()} of {graph.edges.length.toLocaleString()} sketch edges shown · isolated accounts on the outer ring · scroll to zoom, click a node for its explanation</span>
        </div>
      </div>
    </div>
  )
}
