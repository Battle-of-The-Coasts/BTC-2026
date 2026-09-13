import { useEffect, useState } from 'react'
import { api, fmt, pct, LABEL_NAMES } from '../api'
import type { NodeDetail } from '../api'

function Pill({ label }: { label: number }) {
  const cls = label === 1 ? 'bot' : label === 0 ? 'human' : 'ext'
  return <span className={`pill ${cls}`}>{LABEL_NAMES[label]}</span>
}

function SignedBar({ value, max }: { value: number; max: number }) {
  const w = max > 0 ? Math.min(50, (50 * Math.abs(value)) / max) : 0
  const left = value < 0 ? 50 - w : 50
  return <div className="bar"><i style={{ left: `${left}%`, width: `${w}%`, background: value > 0 ? '#e5484d' : '#30a46c', borderRadius: 3 }} /></div>
}

export default function NodeDetails({ ds, idx, onClose, onSelect }: { ds: string; idx: number; onClose: () => void; onSelect: (i: number) => void }) {
  const [d, setD] = useState<NodeDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [showFeatures, setShowFeatures] = useState(false)
  useEffect(() => { setD(null); setErr(null); api.node(ds, idx).then(setD).catch((e) => setErr(String(e))) }, [ds, idx])
  if (err) return <div className="error">{err}</div>
  if (!d) return <div className="loading">Loading node {idx}…</div>
  const n = d.node
  const p = d.propagation
  const chan = Object.entries(p.per_channel).sort((a, b) => Math.abs(b[1].contribution) - Math.abs(a[1].contribution))
  const maxC = Math.max(Math.abs(p.prior_term), ...chan.map(([, c]) => Math.abs(c.contribution)), 1e-6)
  const le = d.local_explanation
  const maxL = le ? Math.max(...le.top.map((t) => Math.abs(t.contribution)), 1e-6) : 1
  const feats = d.features ? Object.entries(d.features).sort((a, b) => Math.abs(b[1].pct - 0.5) - Math.abs(a[1].pct - 0.5)) : []
  return (
    <div>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0 }}>{n.screen_name ? `@${n.screen_name}` : `node ${n.idx}`}</h2>
        <button className="btn small" onClick={onClose}>close</button>
      </div>
      <div className="row small muted" style={{ margin: '4px 0 10px' }}>
        <span>id {n.id}</span>{n.name && <span>· {n.name}</span>}<Pill label={n.label} />{n.subset && <span>· subset {n.subset}</span>}
      </div>
      <div className="card" style={{ marginBottom: 10 }}>
        <div className="title">Trust score (out-of-fold)</div>
        <div className="kpi">{fmt(n.trust)} <small>1 = trusted · 0 = bot-like</small>{n.p_bot != null && <span className={`pill ${n.p_bot > 0.5 ? 'bot' : 'human'}`} style={{ marginLeft: 10, verticalAlign: 'middle' }}>P(bot) {pct(n.p_bot, 0)}</span>}</div>
        <div className="trustbar" style={{ margin: '8px 0' }}><div className="marker" style={{ left: `calc(${100 * n.trust}% - 1px)` }} /></div>
        <div className="grid cols-3 small">
          <div><div className="muted">propagation only</div>{fmt(n.trust_prop_only)}</div>
          <div><div className="muted">local P(bot)</div>{n.p_local == null ? '– (no profile)' : fmt(n.p_local)}</div>
          <div><div className="muted">scored in fold</div>{d.fold}</div>
          <div><div className="muted">followers / following (in data)</div>{n.in_deg} / {n.out_deg}</div>
          <div><div className="muted">mutual</div>{n.mutual_deg}</div>
          <div><div className="muted">community / block</div>{n.community >= 0 ? n.community : '–'} / {n.block >= 0 ? n.block : '–'}</div>
          {n.followers_count != null && <div><div className="muted">profile followers / friends</div>{n.followers_count} / {n.friends_count}</div>}
          {n.statuses_count != null && <div><div className="muted">statuses</div>{n.statuses_count}</div>}
          {n.created_at && <div><div className="muted">created</div>{String(n.created_at).slice(4, 16)}{n.burst && <span className="pill bot" style={{ marginLeft: 4 }}>creation burst</span>}</div>}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 10 }}>
        <div className="title">Why: score decomposition</div>
        <div className="small muted" style={{ marginBottom: 6 }}>
          r = (1−α)·q + α·Σ P<sub>vu</sub> r<sub>u</sub>, trust = ½ − r. α={d.alpha}, λ={d.lam}. Red pushes toward bot, green toward trusted.
        </div>
        <div className="contrib">
          <div>prior (own evidence, q={fmt(p.prior_residual)})</div><SignedBar value={p.prior_term} max={maxC} /><div className="num">{fmt(p.prior_term)}</div>
          {chan.map(([name, c]) => (
            <div key={name} style={{ display: 'contents' }}>
              <div>{name} <span className="muted">({c.n_neighbors} nb · {pct(c.weight_share, 0)} of edges · w={fmt(c.homophily_weight, 2)})</span></div>
              <SignedBar value={c.contribution} max={maxC} /><div>{fmt(c.contribution)}</div>
            </div>
          ))}
          <div><b>residual r</b></div><div />
          <div><b>{fmt(p.residual)}</b></div>
        </div>
        {p.n_neighbors === 0 && <div className="small muted">No neighbours in the graph: the score is the prior only.</div>}
      </div>

      {p.top_neighbors.length > 0 && <div className="card" style={{ marginBottom: 10 }}>
        <div className="title">Most influential neighbours</div>
        <table className="tbl">
          <thead><tr><th>account</th><th>channel</th><th>label</th><th title="neighbour's trust in the fold that scored this node (the value that entered the score)">trust (fold)</th><th>effect</th></tr></thead>
          <tbody>
            {p.top_neighbors.map((nb) => (
              <tr key={`${nb.node}-${nb.channel}`} className="clickable" onClick={() => onSelect(nb.node)}>
                <td>{nb.screen_name ? `@${nb.screen_name}` : nb.id}{nb.is_seed && <span className="muted"> · seed</span>}</td>
                <td className="muted">{nb.channel}</td>
                <td><Pill label={nb.label} /></td>
                <td className="num">{fmt(nb.trust, 2)}</td>
                <td className="num" style={{ color: nb.contribution > 0 ? '#ff8a8e' : '#5fd39a' }}>{nb.contribution > 0 ? '+' : ''}{fmt(nb.contribution, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>}

      {le && <div className="card" style={{ marginBottom: 10 }}>
        <div className="title">Local model: profile &amp; structure evidence</div>
        <div className="small muted" style={{ marginBottom: 6 }}>Random-forest contributions (Saabas). P(bot) = {fmt(le.bias, 2)} (base rate) + Σ contributions.</div>
        <div className="contrib">
          {le.top.map((t) => (
            <div key={t.feature} style={{ display: 'contents' }}>
              <div>{t.feature} <span className="muted">= {fmt(t.value, 2)}</span></div>
              <SignedBar value={t.contribution} max={maxL} /><div>{t.contribution > 0 ? '+' : ''}{fmt(t.contribution, 3)}</div>
            </div>
          ))}
        </div>
      </div>}

      {feats.length > 0 && <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="title" style={{ margin: 0 }}>All features (percentile among labelled accounts)</div>
          <button className="btn small" onClick={() => setShowFeatures(!showFeatures)}>{showFeatures ? 'hide' : 'show'}</button>
        </div>
        {showFeatures && <table className="tbl" style={{ marginTop: 6 }}>
          <tbody>{feats.map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{fmt(v.value, 3)}</td><td className="num muted">p{Math.round(100 * v.pct)}</td></tr>)}</tbody>
        </table>}
      </div>}
    </div>
  )
}
