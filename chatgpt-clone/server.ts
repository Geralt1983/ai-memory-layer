import 'dotenv/config'
import express from 'express'
import bodyParser from 'body-parser'
import fs from 'fs/promises'
import path from 'path'
import crypto from 'crypto'
import fg from 'fast-glob'
import simpleGit from 'simple-git'
import { applyUnifiedPatch } from './src/unified.js'

const app = express()
const HOST = process.env.HOST || '127.0.0.1'
const PORT = Number(process.env.PORT || 7373)
const TOKEN = process.env.AUTH_TOKEN || ''

if (!TOKEN) {
  console.error('Refusing to start: missing AUTH_TOKEN in .env')
  process.exit(1)
}

const REPO_ROOT = process.cwd()
const git = simpleGit(REPO_ROOT)

app.use(bodyParser.json({ limit: '10mb' }))
app.use((req, res, next) => {
  const auth = req.headers.authorization
  if (!auth || !auth.startsWith('Bearer ') || auth.slice(7) !== TOKEN) {
    return res.status(401).json({ error: 'unauthorized' })
  }
  next()
})

function jail(p: string) {
  const abs = path.resolve(REPO_ROOT, p)
  if (!abs.startsWith(REPO_ROOT)) throw new Error('Path escapes repo root')
  return abs
}

async function backup(targetAbs: string) {
  try {
    const data = await fs.readFile(targetAbs)
    const ts = new Date().toISOString().replace(/[:.]/g, '-')
    const bak = `${targetAbs}.${ts}.bak`
    await fs.writeFile(bak, data)
    return bak
  } catch {
    return null
  }
}

app.get('/health', async (_req, res) => {
  const head = await git.revparse(['--short', 'HEAD']).catch(() => '0000000')
  res.json({ ok: true, repo: path.basename(REPO_ROOT), head, cwd: REPO_ROOT })
})

app.get('/fs', async (req, res) => {
  const p = String(req.query.path || '')
  if (!p) return res.status(400).json({ error: 'missing path' })
  try {
    const abs = jail(p)
    const stat = await fs.stat(abs)
    if (stat.isDirectory()) {
      const entries = await fg(['**/*'], { cwd: abs, dot: true, onlyFiles: false, markDirectories: true, deep: 2 })
      return res.json({ type: 'dir', entries })
    } else {
      const content = await fs.readFile(abs, 'utf8')
      const hash = crypto.createHash('sha256').update(content).digest('hex')
      return res.json({ type: 'file', size: stat.size, hash, content })
    }
  } catch (e: any) {
    return res.status(404).json({ error: e.message })
  }
})

app.post('/write', async (req, res) => {
  const { path: rel, content } = req.body || {}
  if (!rel || typeof content !== 'string') return res.status(400).json({ error: 'missing path or content' })
  try {
    const abs = jail(rel)
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await backup(abs)
    await fs.writeFile(abs, content, 'utf8')
    return res.json({ ok: true })
  } catch (e: any) {
    return res.status(500).json({ error: e.message })
  }
})

app.post('/patch', async (req, res) => {
  const body = req.body as { patches: { path: string, unified: string }[] }
  if (!body?.patches?.length) return res.status(400).json({ error: 'missing patches' })
  const results: any[] = []
  for (const p of body.patches) {
    try {
      const abs = jail(p.path)
      await fs.mkdir(path.dirname(abs), { recursive: true })
      const before = (await fs.readFile(abs, 'utf8').catch(() => ''))
      const after = applyUnifiedPatch(before, p.unified, p.path)
      if (after === null) throw new Error('patch failed to apply cleanly')
      await backup(abs)
      await fs.writeFile(abs, after, 'utf8')
      results.push({ path: p.path, ok: true })
    } catch (e: any) {
      results.push({ path: p.path, ok: false, error: e.message })
    }
  }
  const ok = results.every(r => r.ok)
  res.status(ok ? 200 : 207).json({ results })
})

app.get('/git/status', async (_req, res) => {
  const status = await git.status()
  res.json(status)
})

app.post('/git/commit', async (req, res) => {
  const { message, add } = req.body || {}
  if (add) await git.add(add)
  else await git.add(['-A'])
  const r = await git.commit(message || 'chore: local-edit-agent commit')
  res.json({ ok: true, commit: r.commit, summary: r.summary })
})

app.post('/git/branch', async (req, res) => {
  const { name, checkout } = req.body || {}
  if (!name) return res.status(400).json({ error: 'missing name' })
  await git.checkoutLocalBranch(name)
  if (checkout) await git.checkout(name)
  res.json({ ok: true, branch: name })
})

app.listen(PORT, HOST, () => {
  console.log(`Local Edit Agent on http://${HOST}:${PORT} (repo: ${REPO_ROOT})`)
})