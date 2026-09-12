import { useEffect, useState } from 'react'
import { api, fmt, LABEL_NAMES, trustColor } from '../api'
import type { NodeRow, Summary } from '../api'

const COLS: [keyof NodeRow, string][] = [['trust', 'trust'], ['p_bot', 'P(bot)'], ['trust_prop_only', 'prop. only'], ['p_local', 'local P(bot)'], ['in_deg', 'followers'], ['out_deg', 'following'], ['mutual_deg', 'mutual'], ['community', 'community'], ['block', 'block']]

export default function Accounts({ ds, summary, onSelect }: { ds: string; summary: Summary; onSelect: (i: number) => void }) {
  const [q, setQ] = useState('')
  const [label, setLabel] = useState<string>('')
  const [subset, setSubset] = useState('')
  const [sort, setSort] = useState<string>('trust')
  const [order, setOrder] = useState<'asc' | 'desc'>('asc')
  const [page, setPage] = useState(0)
  const [data, setData] = useState<{ total: number; rows: NodeRow[] } | null>(null)
  const limit = 40
  const subsets = Object.keys(summary.dataset).length ? [] : []
  void subsets
  useEffect(() => {
    const t = setTimeout(() => {
      api.nodes(ds, { q, label: label === '' ? undefined : Number(label), subset: subset || undefined, sort, order, limit, offset: page * limit }).then(setData).catch(() => setData({ total: 0, rows: [] }))
    }, 200)
    return () => clearTimeout(t)
  }, [ds, q, label, subset, sort, order, page])
  const toggle = (k: string) => { if (sort === k) setOrder(order === 'asc' ? 'desc' : 'asc'); else { setSort(k); setOrder('asc') }; setPage(0) }
  const pages = data ? Math.ceil(data.total / limit) : 0
  return (
    <div className="card">
      <div className="row" style={{ marginBottom: 10 }}>
        <input placeholder="search id or @handle" value={q} onChange={(e) => { setQ(e.target.value); setPage(0) }} style={{ width: 220 }} />
        <select value={label} onChange={(e) => { setLabel(e.target.value); setPage(0) }}><option value="">all labels</option><option value="1">bots</option><option value="0">humans</option></select>
        <input placeholder="subset (e.g. FSF)" value={subset} onChange={(e) => { setSubset(e.target.value); setPage(0) }} style={{ width: 130 }} />
        <span className="muted small">{data ? `${data.total.toLocaleString()} accounts` : '…'}</span>
        <span style={{ marginLeft: 'auto' }} className="row small">
          <button className="btn small" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button> page {page + 1} / {Math.max(1, pages)} <button className="btn small" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>›</button>
        </span>
      </div>
      <table className="tbl">
        <thead><tr><th>account</th><th>label</th><th>subset</th>{COLS.map(([k, n]) => <th key={k} onClick={() => toggle(k)}>{n}{sort === k ? (order === 'asc' ? ' ▲' : ' ▼') : ''}</th>)}<th>burst</th></tr></thead>
        <tbody>
          {data?.rows.map((r) => (
            <tr key={r.idx} className="clickable" onClick={() => onSelect(r.idx)}>
              <td>{r.screen_name ? `@${r.screen_name}` : r.id}</td>
              <td><span className={`pill ${r.label === 1 ? 'bot' : r.label === 0 ? 'human' : 'ext'}`}>{LABEL_NAMES[r.label]}</span></td>
              <td className="muted">{r.subset}</td>
              <td className="num"><span style={{ color: trustColor(r.trust) }}>{fmt(r.trust)}</span></td>
              <td className="num">{r.p_bot == null ? '–' : fmt(r.p_bot, 2)}</td>
              <td className="num">{fmt(r.trust_prop_only)}</td><td className="num">{r.p_local == null ? '–' : fmt(r.p_local)}</td>
              <td className="num">{r.in_deg}</td><td className="num">{r.out_deg}</td><td className="num">{r.mutual_deg}</td>
              <td className="num">{r.community >= 0 ? r.community : '–'}</td><td className="num">{r.block >= 0 ? r.block : '–'}</td>
              <td>{r.burst ? '●' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
