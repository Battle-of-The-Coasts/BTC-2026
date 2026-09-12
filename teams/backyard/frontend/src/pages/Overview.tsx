import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { METHOD_COLORS, METHOD_LABELS, fmt, pct } from '../api'
import type { Clusters, Metrics, Summary } from '../api'

const ORDER = ['trust_score', 'trust_score_uncal', 'trust_score_fixed_w', 'propagation_only', 'local_rf', 'sgc', 'trustrank']

export default function Overview({ summary, metrics, clusters }: { summary: Summary; metrics: Metrics; clusters: Clusters }) {
  const cf = metrics.crossfit
  const ts = cf.trust_score
  const rocSeries = Object.entries(cf).map(([m, v]) => ({ m, points: v.roc.map(([x, y]) => ({ x, y })) }))
  const hist = metrics.histograms.bots.edges.slice(0, -1).map((e, i) => ({
    bin: `${e.toFixed(2)}`, bots: metrics.histograms.bots.counts[i], humans: metrics.histograms.humans.counts[i],
  }))
  const fracs = Array.from(new Set(metrics.seed_sweep.map((r) => r.seed_frac))).sort((a, b) => (a as number) - (b as number)) as number[]
  const methodsSweep = ORDER.filter((m) => metrics.seed_sweep.some((r) => r.method === m))
  const sweepData = fracs.map((f) => {
    const row: Record<string, number | string> = { frac: `${(100 * f).toFixed(f < 0.05 ? 0 : 0)}%` }
    for (const m of methodsSweep) { const r = metrics.seed_sweep.find((x) => x.method === m && x.seed_frac === f); if (r) row[m] = +r.auc.toFixed(4) }
    return row
  })
  const noises = Array.from(new Set(metrics.noise_sweep.map((r) => r.noise))).sort((a, b) => (a as number) - (b as number)) as number[]
  const noiseData = noises.map((n) => {
    const row: Record<string, number | string> = { noise: `${Math.round(100 * n)}%` }
    for (const m of methodsSweep) { const r = metrics.noise_sweep.find((x) => x.method === m && x.noise === n); if (r) row[m] = +r.auc.toFixed(4) }
    return row
  })
  const imp = metrics.feature_importance.permutation_auc ?? metrics.feature_importance.impurity
  const impRows = Object.entries(imp).sort((a, b) => b[1] - a[1]).slice(0, 15).map(([f, v]) => ({ f, v: +v.toFixed(4) }))
  const w0 = metrics.channel_weights['0'] ?? {}
  const sel = metrics.selected_params?.['0']
  const hom = metrics.homophily
  const pc = summary.config.propagation
  const dsn = summary.dataset
  const sus = clusters.communities.filter((c) => c.predicted_bot_share > 0.5).length
  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="grid cols-4">
        <div className="card"><div className="title">Accounts</div><div className="kpi">{dsn.n_labeled.toLocaleString()}<small>labelled of {dsn.n_nodes.toLocaleString()} nodes</small></div>
          <div className="small muted">{dsn.n_bots.toLocaleString()} bots · {dsn.n_humans.toLocaleString()} humans</div></div>
        <div className="card"><div className="title">Trust score · ROC AUC</div><div className="kpi">{fmt(ts.auc, 4)}<small>out-of-fold, {summary.crossfit_folds} folds</small></div>
          <div className="small muted">AP {fmt(ts.ap, 4)} · F1 {fmt(ts.f1, 3)} @ calibrated P(bot)&gt;0.5</div></div>
        <div className="card"><div className="title">Edges / channels</div><div className="kpi">{Object.values(dsn.relations as Record<string, number>).reduce((a, b) => a + b, 0).toLocaleString()}<small>relation edges</small></div>
          <div className="small muted">{Object.keys(w0).length} propagation channels · α={sel?.alpha ?? pc.alpha}, λ={sel?.lam ?? pc.lam} (selected on seeds)</div></div>
        <div className="card"><div className="title">Clusters</div><div className="kpi">{clusters.n_communities}<small>Leiden communities</small></div>
          <div className="small muted">{sus} with predicted bot share &gt; 50% · {clusters.dense_blocks.length} dense blocks</div></div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h2>Method comparison (out-of-fold, every labelled account scored without its label)</h2>
          <div className="small muted" style={{ marginBottom: 6 }}>F1 / precision / recall at P(bot) &gt; 0.5 on each method's probability; TrustRank is ranking-only (AUC/AP).</div>
          <table className="tbl">
            <thead><tr><th>method</th><th>AUC</th><th>AP</th><th>F1</th><th>prec.</th><th>recall</th><th>acc.</th></tr></thead>
            <tbody>
              {ORDER.filter((m) => cf[m]).map((m) => (
                <tr key={m}><td><span className="sw" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 5, background: METHOD_COLORS[m], marginRight: 6 }} />{METHOD_LABELS[m] ?? m}</td>
                  <td className="num">{fmt(cf[m].auc, 4)}</td><td className="num">{fmt(cf[m].ap, 4)}</td><td className="num">{fmt(cf[m].f1, 3)}</td>
                  <td className="num">{fmt(cf[m].precision, 3)}</td><td className="num">{fmt(cf[m].recall, 3)}</td><td className="num">{fmt(cf[m].accuracy, 3)}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 8 }}>Per collection subset (mean trust · share flagged as bot):{' '}
            {Object.entries(metrics.per_subset).map(([s, v]) => <span key={s} style={{ marginRight: 10 }}><b>{s}</b> {v.label === 1 ? '(bots)' : v.label === 0 ? '(humans)' : ''} {fmt(v.mean_trust, 2)} · {pct(v.flagged_share, 0)}</span>)}
          </div>
        </div>
        <div className="card">
          <h2>ROC curves</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid stroke="#2a2f3d" />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fill: '#9a9fb3', fontSize: 11 }} label={{ value: 'false positive rate', fill: '#9a9fb3', fontSize: 11, dy: 12 }} />
              <YAxis dataKey="y" type="number" domain={[0, 1]} tick={{ fill: '#9a9fb3', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1f2330', border: '1px solid #2a2f3d' }} />
              {rocSeries.map((s) => <Line key={s.m} data={s.points} dataKey="y" name={METHOD_LABELS[s.m] ?? s.m} stroke={METHOD_COLORS[s.m] ?? '#fff'} dot={false} isAnimationActive={false} strokeWidth={s.m === 'trust_score' ? 2.5 : 1.2} />)}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid cols-3">
        <div className="card">
          <h2>Raw formula trust (½ − r) by label</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hist}>
              <CartesianGrid stroke="#2a2f3d" /><XAxis dataKey="bin" tick={{ fill: '#9a9fb3', fontSize: 10 }} /><YAxis tick={{ fill: '#9a9fb3', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2330', border: '1px solid #2a2f3d' }} />
              <Bar dataKey="bots" fill="#e5484d" isAnimationActive={false} /><Bar dataKey="humans" fill="#30a46c" isAnimationActive={false} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h2>AUC vs. share of labelled seeds</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={sweepData}>
              <CartesianGrid stroke="#2a2f3d" /><XAxis dataKey="frac" tick={{ fill: '#9a9fb3', fontSize: 10 }} /><YAxis domain={['auto', 1]} tick={{ fill: '#9a9fb3', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2330', border: '1px solid #2a2f3d' }} />
              {methodsSweep.map((m) => <Line key={m} dataKey={m} name={METHOD_LABELS[m] ?? m} stroke={METHOD_COLORS[m]} isAnimationActive={false} strokeWidth={m === 'trust_score' ? 2.5 : 1.2} />)}
            </LineChart>
          </ResponsiveContainer>
          <div className="small muted">mean over 5 random seed draws; evaluation on the non-seed accounts</div>
        </div>
        <div className="card">
          <h2>AUC vs. label noise in seeds (10% seeds)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={noiseData}>
              <CartesianGrid stroke="#2a2f3d" /><XAxis dataKey="noise" tick={{ fill: '#9a9fb3', fontSize: 10 }} /><YAxis domain={['auto', 1]} tick={{ fill: '#9a9fb3', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1f2330', border: '1px solid #2a2f3d' }} />
              {methodsSweep.map((m) => <Line key={m} dataKey={m} name={METHOD_LABELS[m] ?? m} stroke={METHOD_COLORS[m]} isAnimationActive={false} strokeWidth={m === 'trust_score' ? 2.5 : 1.2} />)}
            </LineChart>
          </ResponsiveContainer>
          <div className="small muted">seed labels flipped at random before training / propagation</div>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h2>Local model: most useful features (permutation importance, ΔAUC)</h2>
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={impRows} layout="vertical" margin={{ left: 120 }}>
              <CartesianGrid stroke="#2a2f3d" /><XAxis type="number" tick={{ fill: '#9a9fb3', fontSize: 10 }} /><YAxis type="category" dataKey="f" width={120} tick={{ fill: '#e6e8ef', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1f2330', border: '1px solid #2a2f3d' }} />
              <Bar dataKey="v" fill="#4cc9f0" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
          <div className="small muted">feature blocks: {Object.entries(summary.feature_blocks).map(([k, v]) => `${k} (${v})`).join(' · ')}</div>
        </div>
        <div className="card">
          <h2>Propagation channels (fold 0): homophily measured on seed–seed edges</h2>
          <table className="tbl">
            <thead><tr><th>channel</th><th>seed edges</th><th>same-label</th><th>φ</th><th>prior w</th><th>used w</th></tr></thead>
            <tbody>
              {Object.entries(hom).filter(([k]) => !k.startsWith('_')).map(([k, v]: [string, any]) => (
                <tr key={k}><td>{k}</td><td className="num">{v.seed_edges}</td><td className="num">{fmt(v.same_label_rate, 3)}</td><td className="num">{fmt(v.phi ?? v.kappa, 3)}</td><td className="num">{fmt(v.prior, 2)}</td><td className="num"><b>{fmt(v.weight, 3)}</b></td></tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 6 }}>same-label rate under independence: {fmt(hom._independence_same_rate, 3)} · propagation converged in {metrics.propagation_info.iterations} iterations ·
            hub damping {pc.hub_damping} · pipeline {summary.timings.total}s</div>
          {sel && <div className="small" style={{ marginTop: 6 }}>Selected by inner cross-validation on the seeds (fold 0): <b>α = {sel.alpha}</b>, <b>λ = {sel.lam}</b>; calibration P(bot) = σ({fmt(sel.platt_a, 1)}·r {sel.platt_b >= 0 ? '+' : '−'} {fmt(Math.abs(sel.platt_b), 2)}) fitted on {sel.calibrated_on}.
            <span className="muted"> Grid: {Object.entries(sel.grid).map(([k, g]) => `${k}: ${g.auc.toFixed(3)}`).join(' · ')}</span></div>}
        </div>
      </div>
    </div>
  )
}
