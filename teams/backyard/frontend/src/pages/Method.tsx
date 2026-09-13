import { useEffect, useState } from 'react'
import { marked } from 'marked'
import { api } from '../api'

export default function Method() {
  const [html, setHtml] = useState<string>('')
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { api.docs().then(async (md) => setHtml(await marked.parse(md))).catch((e) => setErr(String(e))) }, [])
  if (err) return <div className="error">{err}</div>
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
}
