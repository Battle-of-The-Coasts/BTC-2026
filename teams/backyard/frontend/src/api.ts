export type DatasetInfo = { name: string; summary: Summary }
export type Summary = {
  dataset: Record<string, any>
  config: { propagation: { alpha: number; lam: number; hub_damping: number; relation_weights: Record<string, number>; seed_residual: number }; [k: string]: any }
  timings: Record<string, number>
  n_display_nodes: number
  n_display_edges: number
  n_features: number
  feature_blocks: Record<string, number>
  crossfit_folds: number
}
export type MetricRow = { n: number; auc: number; ap: number; f1: number; precision: number; recall: number; accuracy: number }
export type SweepRow = MetricRow & { method: string; seed_frac?: number; noise?: number; auc_std: number; ap_std: number; f1_std: number; n_seeds?: number }
export type Metrics = {
  crossfit: Record<string, MetricRow & { roc: [number, number][] }>
  per_subset: Record<string, { n: number; label: number | null; mean_trust: number; flagged_share: number }>
  seed_sweep: SweepRow[]
  noise_sweep: SweepRow[]
  histograms: { bots: Hist; humans: Hist; external: Hist | null }
  feature_importance: { impurity: Record<string, number>; permutation_auc?: Record<string, number> }
  feature_blocks: Record<string, string[]>
  channel_weights: Record<string, Record<string, number>>
  homophily: Record<string, any>
  selected_params?: Record<string, { alpha: number; lam: number; platt_a: number; platt_b: number; calibrated_on: string; grid: Record<string, { auc: number; ap: number }> }>
  propagation_info: { iterations: number; converged: boolean }
}
export type Hist = { counts: number[]; edges: number[] }
export type GraphNode = { i: number; x: number; y: number; t: number; p?: number; l: number; s: string; c: number; b: number; d: number; id: string; n?: string }
export type GraphData = { nodes: GraphNode[]; edges: [number, number, string, number][]; edge_types: string[] }
export type Community = {
  id: number; size: number; mean_trust: number; min_trust: number; predicted_bot_share: number; true_bot_share: number
  internal_density: number; subsets: Record<string, number>; burst_share: number | null; members_preview: number[]
  top_targets?: { node: number; followed_by: number; share: number }[]
}
export type DenseBlock = {
  id: number; size: number; n_targets: number; density: number; mean_trust: number; predicted_bot_share: number
  true_bot_share: number; subsets: Record<string, number>; members_preview: number[]; top_targets: number[]
}
export type Clusters = {
  mode: string; resolution: number; n_communities: number; communities: Community[]; dense_blocks: DenseBlock[]
  burst_days: { day: string; n_accounts: number }[]; burst_min_accounts: number
}
export type NodeRow = {
  idx: number; id: string; label: number; subset: string; trust: number; p_bot?: number; trust_prop_only: number; p_local: number | null
  in_deg: number; out_deg: number; mutual_deg: number; community: number; block: number; burst: boolean; fold: number
  screen_name?: string | null; name?: string | null; created_at?: string | null
  followers_count?: number | null; friends_count?: number | null; statuses_count?: number | null
}
export type NodeDetail = {
  node: NodeRow
  fold: number
  alpha: number
  lam: number
  weights: Record<string, number>
  community: number | null
  propagation: {
    residual: number; trust: number; prior_residual: number; prior_term: number; neighbor_term: number; n_neighbors: number
    p_bot?: number
    per_channel: Record<string, { contribution: number; n_neighbors: number; weight_share: number; homophily_weight?: number }>
    top_neighbors: { node: number; contribution: number; weight: number; channel: string; label: number; trust: number; trust_oof?: number; id: string; screen_name: string | null; is_seed: boolean }[]
  }
  features: Record<string, { value: number; pct: number }> | null
  local_explanation: { bias: number; top: { feature: string; contribution: number; value: number }[] } | null
}
export type WhatIfResult = { alpha: number; lam: number; weights: Record<string, number>; fold: number; iterations: number; metrics: MetricRow; trust: Record<string, number>; p_bot?: Record<string, number> }

const base = '/api'

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${url}`)
  return r.json() as Promise<T>
}

export const api = {
  datasets: () => getJSON<DatasetInfo[]>(`${base}/datasets`),
  summary: (ds: string) => getJSON<Summary>(`${base}/${ds}/summary`),
  metrics: (ds: string) => getJSON<Metrics>(`${base}/${ds}/metrics`),
  graph: (ds: string) => getJSON<GraphData>(`${base}/${ds}/graph`),
  clusters: (ds: string) => getJSON<Clusters>(`${base}/${ds}/clusters`),
  clusterMembers: (ds: string, id: number, kind: 'community' | 'block') =>
    getJSON<{ id: number; members: number[] }>(`${base}/${ds}/cluster/${id}/members?kind=${kind}`),
  nodes: (ds: string, params: Record<string, string | number | boolean | undefined>) => {
    const q = Object.entries(params).filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join('&')
    return getJSON<{ total: number; rows: NodeRow[] }>(`${base}/${ds}/nodes?${q}`)
  },
  node: (ds: string, idx: number) => getJSON<NodeDetail>(`${base}/${ds}/node/${idx}`),
  whatif: async (ds: string, body: { alpha?: number; lam?: number; weights?: Record<string, number>; fold?: number }) => {
    const r = await fetch(`${base}/${ds}/whatif`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    if (!r.ok) throw new Error(`whatif failed: ${r.status}`)
    return r.json() as Promise<WhatIfResult>
  },
  docs: async () => {
    const r = await fetch(`${base}/docs`)
    if (!r.ok) throw new Error('docs unavailable')
    return r.text()
  },
}

export const METHOD_LABELS: Record<string, string> = {
  trust_score: 'Trust score (prior + propagation, learned weights)',
  trust_score_fixed_w: 'Trust score (fixed channel weights)',
  trust_score_uncal: 'Trust score, raw rule trust < ½ (no calibration)',
  propagation_only: 'Propagation only (seeds, no features)',
  local_rf: 'Local model only (Random Forest)',
  trustrank: 'TrustRank / SybilRank baseline',
  sgc: 'SGC baseline (graph convolution + LR)',
}
export const METHOD_COLORS: Record<string, string> = {
  trust_score: '#4cc9f0', trust_score_fixed_w: '#90e0ef', trust_score_uncal: '#2a7f9e', propagation_only: '#f9c74f', local_rf: '#f8961e', trustrank: '#b5b5c3', sgc: '#c77dff',
}

export function trustColor(t: number): string {
  // 0 (untrusted) red -> 0.5 amber -> 1 (trusted) green
  const x = Math.max(0, Math.min(1, t))
  const stops: [number, [number, number, number]][] = [[0, [229, 72, 77]], [0.5, [245, 183, 49]], [1, [48, 164, 108]]]
  let a = stops[0], b = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) if (x >= stops[i][0] && x <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break }
  const f = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0])
  const c = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}
export const LABEL_COLORS: Record<number, string> = { 1: '#e5484d', 0: '#30a46c', [-1]: '#6f7285' }
export const LABEL_NAMES: Record<number, string> = { 1: 'bot / fake', 0: 'human', [-1]: 'unlabelled (external)' }
export const PALETTE = ['#4cc9f0', '#f72585', '#b5179e', '#7209b7', '#4361ee', '#4895ef', '#f9c74f', '#f8961e', '#90be6d', '#43aa8b', '#577590', '#f94144', '#ff7f50', '#9b5de5', '#00f5d4', '#fee440']
export function catColor(i: number): string { return i < 0 ? '#6f7285' : PALETTE[i % PALETTE.length] }
export const fmt = (x: number | null | undefined, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? '–' : x.toFixed(d))
export const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '–' : `${(100 * x).toFixed(d)}%`)
