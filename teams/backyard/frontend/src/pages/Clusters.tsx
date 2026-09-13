import { useState } from 'react'
import { fmt, pct, trustColor } from '../api'
import type { Clusters } from '../api'
import type { Highlight } from '../App'

export default function ClustersPage({ ds, clusters, onSelect, onHighlight }: { ds: string; clusters: Clusters; onSelect: (i: number) => void; onHighlight: (h: Highlight) => void }) {
  const [sort, setSort] = useState<'size' | 'predicted_bot_share' | 'mean_trust' | 'true_bot_share'>('size')
  const [minSize, setMinSize] = useState(5)
  const comms = [...clusters.communities].filter((c) => c.size >= minSize).sort((a, b) => (b[sort] as number) - (a[sort] as number))
  const hidden = clusters.communities.length - comms.length
  void ds
  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="card">
        <h2>Leiden communities on the cluster graph</h2>
        <div className="small muted" style={{ marginBottom: 8 }}>graph: {clusters.mode} · resolution {clusters.resolution} · {clusters.n_communities} communities.
          “Predicted” uses the trust score only; “ground truth” uses the dataset labels and is shown for evaluation.</div>
        <div className="row small" style={{ marginBottom: 6 }}>sort by
          {(['size', 'predicted_bot_share', 'mean_trust', 'true_bot_share'] as const).map((k) => <button key={k} className={`btn small ${sort === k ? 'active' : ''}`} onClick={() => setSort(k)}>{k.replace(/_/g, ' ')}</button>)}
          <span className="muted" style={{ marginLeft: 12 }}>min size</span>
          <input type="number" min={1} value={minSize} onChange={(e) => setMinSize(Math.max(1, Number(e.target.value)))} style={{ width: 60 }} />
          {hidden > 0 && <span className="muted">({hidden} smaller communities hidden — mostly isolated accounts that form singleton communities)</span>}
        </div>
        <table className="tbl">
          <thead><tr><th>#</th><th>size</th><th>mean trust</th><th>predicted bot share</th><th>ground-truth bot share</th><th>density</th><th>creation burst</th><th>subsets</th><th>most-followed targets</th><th></th></tr></thead>
          <tbody>
            {comms.map((c) => (
              <tr key={c.id}>
                <td>{c.id}</td><td className="num">{c.size}</td>
                <td className="num"><span style={{ color: trustColor(c.mean_trust) }}>{fmt(c.mean_trust, 3)}</span></td>
                <td className="num">{pct(c.predicted_bot_share, 0)}</td><td className="num muted">{pct(c.true_bot_share, 0)}</td>
                <td className="num">{fmt(c.internal_density, 3)}</td><td className="num">{c.burst_share == null ? '–' : pct(c.burst_share, 0)}</td>
                <td className="small">{Object.entries(c.subsets).map(([s, n]) => `${s}:${n}`).join(' ')}</td>
                <td className="small muted">{(c.top_targets ?? []).slice(0, 3).map((t) => <span key={t.node} className="clickable" style={{ marginRight: 6, cursor: 'pointer' }} onClick={() => onSelect(t.node)}>{t.node} ({pct(t.share, 0)})</span>)}</td>
                <td><button className="btn small" onClick={() => onHighlight({ kind: 'community', id: c.id })}>show in network</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid cols-2">
        <div className="card">
          <h2>Dense co-following blocks (Fraudar-style greedy peeling)</h2>
          <div className="small muted" style={{ marginBottom: 8 }}>groups of accounts that follow the same targets in lockstep; density = edge weight per node. Dense ≠ bot: the trust score says whether the group is trusted.</div>
          <table className="tbl">
            <thead><tr><th>#</th><th>accounts</th><th>targets</th><th>density</th><th>mean trust</th><th>predicted bot</th><th>ground truth</th><th>subsets</th><th></th></tr></thead>
            <tbody>
              {clusters.dense_blocks.map((b) => (
                <tr key={b.id}><td>{b.id}</td><td className="num">{b.size}</td><td className="num">{b.n_targets}</td><td className="num">{fmt(b.density, 2)}</td>
                  <td className="num" style={{ color: trustColor(b.mean_trust) }}>{fmt(b.mean_trust, 3)}</td><td className="num">{pct(b.predicted_bot_share, 0)}</td><td className="num muted">{pct(b.true_bot_share, 0)}</td>
                  <td className="small">{Object.entries(b.subsets).map(([s, n]) => `${s}:${n}`).join(' ')}</td>
                  <td><button className="btn small" onClick={() => onHighlight({ kind: 'block', id: b.id })}>show</button></td></tr>
              ))}
              {clusters.dense_blocks.length === 0 && <tr><td colSpan={9} className="muted">no follow relation in this dataset</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h2>Account-creation bursts</h2>
          <div className="small muted" style={{ marginBottom: 8 }}>days on which ≥ {clusters.burst_min_accounts} of the labelled accounts were created (bulk registration is typical of purchased accounts)</div>
          {clusters.burst_days.length === 0 && <div className="muted small">no creation dates in this dataset</div>}
          <div className="row">
            {clusters.burst_days.slice(0, 40).map((d) => <span key={d.day} className="pill ext">{d.day} · {d.n_accounts}</span>)}
          </div>
        </div>
      </div>
    </div>
  )
}
