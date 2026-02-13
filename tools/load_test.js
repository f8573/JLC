#!/usr/bin/env node
/**
 * Simple load tester: sends many 512x512 matrices to the diagnostics API.
 * Usage: node tools/load_test.js [--url http://localhost:8080/api/diagnostics] [--requests 100] [--concurrency 10]
 */

const DEFAULT_URL = 'http://localhost:8080/api/diagnostics'
const DEFAULT_REQUESTS = 100
const DEFAULT_CONCURRENCY = 10

function parseArgs() {
  const args = process.argv.slice(2)
  const opts = { url: DEFAULT_URL, requests: DEFAULT_REQUESTS, concurrency: DEFAULT_CONCURRENCY, benchmark: false }
  for (let i = 0; i < args.length; i++) {
    const a = args[i]
    if (a === '--url' && args[i + 1]) { opts.url = args[++i]; continue }
    if (a === '--requests' && args[i + 1]) { opts.requests = Number(args[++i]) || opts.requests; continue }
    if (a === '--concurrency' && args[i + 1]) { opts.concurrency = Number(args[++i]) || opts.concurrency; continue }
    if (a === '--benchmark') { opts.benchmark = true; continue }
  }
  return opts
}

function generateMatrix(n = 512) {
  const mat = new Array(n)
  for (let i = 0; i < n; i++) {
    const row = new Array(n)
    for (let j = 0; j < n; j++) row[j] = Math.random()
    mat[i] = row
  }
  return mat
}

async function sendOne(url, matrix, id) {
  const body = JSON.stringify({ matrix })
  const start = Date.now()
  try {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
    const text = await res.text()
    const duration = Date.now() - start
    console.log(`#${id} ${res.status} ${res.statusText} ${duration}ms`)
    return { ok: res.ok, status: res.status, body: text }
  } catch (err) {
    const duration = Date.now() - start
    console.error(`#${id} ERROR ${err.message} ${duration}ms`)
    return { ok: false, error: String(err) }
  }
}

async function run() {
  const opts = parseArgs()
  const url = opts.url
  const requests = opts.requests
  const concurrency = opts.concurrency
  console.log(`Load test: ${requests} requests -> ${url} (concurrency=${concurrency}) benchmark=${opts.benchmark}`)

  // Node 18+ provides global fetch; verify
  if (typeof fetch !== 'function') {
    console.error('This script requires Node 18+ (global fetch).')
    process.exit(1)
  }

  const matrix = generateMatrix(512)

  let inFlight = 0
  let launched = 0
  let completed = 0

  const results = []

  return new Promise((resolve) => {
    function tick() {
      while (inFlight < concurrency && launched < requests) {
        launched++
        inFlight++
        const id = launched
        if (opts.benchmark) {
          const start = Date.now()
          fetch(url, { method: 'GET' }).then(async (res) => {
            const text = await res.text()
            const duration = Date.now() - start
            console.log(`#${id} ${res.status} ${res.statusText} ${duration}ms`)
            results.push({ ok: res.ok, status: res.status, body: text })
          }).catch((err) => {
            const duration = Date.now() - start
            console.error(`#${id} ERROR ${err.message} ${duration}ms`)
            results.push({ ok: false, error: String(err) })
          }).finally(() => {
            inFlight--
            completed++
            if (completed % 10 === 0 || completed === requests) console.log(`Completed ${completed}/${requests}`)
            if (completed === requests) resolve(results)
            else tick()
          })
        } else {
          sendOne(url, matrix, id).then((r) => {
            results.push(r)
          }).catch((e) => {
            results.push({ ok: false, error: String(e) })
          }).finally(() => {
            inFlight--
            completed++
            if (completed % 10 === 0 || completed === requests) console.log(`Completed ${completed}/${requests}`)
            if (completed === requests) resolve(results)
            else tick()
          })
        }
      }
    }
    tick()
  })
}

if (require.main === module) {
  run().then((res) => {
    const ok = res.filter(r => r && r.ok).length
    console.log(`Finished: ${ok}/${res.length} succeeded`)
  }).catch((e) => {
    console.error('Run failed:', e)
    process.exit(2)
  })
}

module.exports = { run, generateMatrix }
