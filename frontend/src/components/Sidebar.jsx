import React, { useEffect, useState, useRef } from 'react'

/**
 * Unified navigation sidebar across all pages.
 *
 * Structure:
 * - Libraries
 *   - Compute
 *     - Current Analysis (only visible on /matrix routes)
 *   - Favorites
 *   - History
 * - Recent Sessions
 *
 * @param {Object} props
 * @param {string} [props.active='home'] - Active sidebar item
 * @param {boolean} [props.showCurrentAnalysis=false] - Whether to show Current Analysis link
 */
export default function Sidebar({ active = 'home', showCurrentAnalysis = false }) {
  const [sessions, setSessions] = useState([])
  const [computeExpanded, setComputeExpanded] = useState(true)
  const [diagnostics, setDiagnostics] = useState(null)
  const [running, setRunning] = useState(false)
  const [benchmarkRunning, setBenchmarkRunning] = useState(false)
  const [benchmarkProgress, setBenchmarkProgress] = useState({ complete: 0, total: 0, phase: '' })
  const [maxGflops, setMaxGflops] = useState(null)
  const [recentGflops, setRecentGflops] = useState(null)
  const [barWidthPercent, setBarWidthPercent] = useState(45)
  const [barColorClass, setBarColorClass] = useState('bg-primary')
  const [showInfoModal, setShowInfoModal] = useState(false)
  const initialBenchmarkDone = useRef(false)
  const benchmarkIntervalRef = useRef(null)

  useEffect(() => {
    function load() {
      try {
        const raw = localStorage.getItem('recentSessions')
        const arr = raw ? JSON.parse(raw) : []
        setSessions(arr.slice(0, 5)) // Show only 5 recent sessions in sidebar
      } catch (e) {
        setSessions([])
      }
    }

    // initial load and listen for storage changes
    load()
    window.addEventListener('storage', load)
    return () => window.removeEventListener('storage', load)
  }, [])

  // Fetch current backend status snapshot.
  async function fetchDiagnostics() {
    setRunning(true)
    try {
      const res = await fetch('/api/status')
      if (!res.ok) throw new Error('no-status')
      const json = await res.json()
      setDiagnostics(json)
      // If CPU is online and we haven't run initial benchmark, run it
      const cpuOnline = json && json.cpu && String(json.cpu.state).toLowerCase() === 'online'
      if (cpuOnline && !initialBenchmarkDone.current) {
        initialBenchmarkDone.current = true
        // run initial benchmark but account it as queued job
        runSystemBenchmark()
      }
    } catch (e) {
      setDiagnostics({ status: 'SERVICE_INTERRUPTION', cpu: { name: 'CPU', gflops: null, state: 'offline', queuedJobs: 0 } })
    } finally {
      setRunning(false)
    }
  }

  useEffect(() => {
    // attempt to load diagnostics on mount
    fetchDiagnostics()
    // open SSE stream for status updates (only apply when changed)
    const es = new EventSource('/api/diagnostics/stream')
    es.addEventListener('status', (ev) => {
      try {
        const data = JSON.parse(ev.data)
        setDiagnostics(data)
        // start initial benchmark if CPU becomes online via SSE
        const cpuOnline = data && data.cpu && String(data.cpu.state).toLowerCase() === 'online'
        if (cpuOnline && !initialBenchmarkDone.current) {
          initialBenchmarkDone.current = true
          runSystemBenchmark()
        }
      } catch (e) {
        // ignore malformed events
      }
    })
    es.onerror = () => {
      // if stream fails, close and leave current diagnostics as-is
      try { es.close() } catch (e) {}
    }
    return () => { try { es.close() } catch (e) {} }
  }, [])

  // schedule periodic benchmarks every 5 minutes
  useEffect(() => {
    // runSystemBenchmark is stable within this component scope
    // schedule interval
    benchmarkIntervalRef.current = setInterval(() => {
      // don't start a new one if one is running
      if (!benchmarkRunning) runSystemBenchmark()
    }, 5 * 60 * 1000)
    return () => {
      if (benchmarkIntervalRef.current) clearInterval(benchmarkIntervalRef.current)
    }
  }, [benchmarkRunning])

  const cpuState = diagnostics && diagnostics.cpu && diagnostics.cpu.state ? String(diagnostics.cpu.state).toLowerCase() : null
  const queuedJobs = diagnostics && diagnostics.cpu && Number.isFinite(Number(diagnostics.cpu.queuedJobs))
    ? Number(diagnostics.cpu.queuedJobs)
    : 0

  // Determine if the current user has a pending submitted job and its queued position.
  let userPendingJobPos = null
  try {
    if (window && window.__myPendingJob && Number.isFinite(Number(window.__myPendingJob.position))) {
      userPendingJobPos = Number(window.__myPendingJob.position)
    }
  } catch {}

  const systemStatus = (() => {
    if (cpuState !== 'online') return 'SERVICE_INTERRUPTION'
    // If user has submitted a job, use position-specific thresholds
    if (userPendingJobPos !== null) {
      if (userPendingJobPos >= 101) return 'LARGE_QUEUE'
      if (userPendingJobPos >= 11) return 'BUSY'
      return 'ONLINE'
    }
    // No user-submitted job: use global queue thresholds
    if (queuedJobs >= 100) return 'LARGE_QUEUE'
    if (queuedJobs >= 10) return 'BUSY'
    return 'ONLINE'
  })()
  const cpuColor = cpuState === 'online' ? 'emerald' : (cpuState === 'offline' ? 'rose' : 'amber')
  const cpuPillClass = cpuColor === 'emerald'
    ? 'inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-100 dark:border-emerald-500/30 text-emerald-700 dark:text-emerald-300 text-sm font-medium'
    : (cpuColor === 'rose'
      ? 'inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-rose-50 dark:bg-rose-500/10 border border-rose-100 dark:border-rose-500/30 text-rose-700 dark:text-rose-300 text-sm font-medium'
      : 'inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-amber-50 dark:bg-amber-500/10 border border-amber-100 dark:border-amber-500/30 text-amber-700 dark:text-amber-300 text-sm font-medium')
  const cpuPingClass = cpuColor === 'emerald' ? 'animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75'
    : (cpuColor === 'rose' ? 'animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75'
      : 'animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75')
  const cpuDotClass = cpuColor === 'emerald' ? 'relative inline-flex rounded-full h-2 w-2 bg-emerald-500'
    : (cpuColor === 'rose' ? 'relative inline-flex rounded-full h-2 w-2 bg-rose-500'
      : 'relative inline-flex rounded-full h-2 w-2 bg-amber-500')

  const BENCHMARK_SIZES = [512]
  const BENCHMARK_ITERATIONS = 5
  const BENCHMARK_TEST = 'GEMM'
  const BENCHMARK_PARTS = [
    { label: 'determinant', present: (p) => p?.basicProperties?.scalarInvariants?.determinant !== undefined && p?.basicProperties?.scalarInvariants?.determinant !== null },
    { label: 'inverse', present: (p) => !!p?.matrixDecompositions?.derivedMatrices?.inverse },
    { label: 'qr', present: (p) => !!p?.matrixDecompositions?.primaryDecompositions?.qr },
    { label: 'lu', present: (p) => !!p?.matrixDecompositions?.primaryDecompositions?.lu },
    { label: 'cholesky', present: (p) => !!p?.matrixDecompositions?.primaryDecompositions?.cholesky },
    { label: 'svd', present: (p) => !!p?.matrixDecompositions?.primaryDecompositions?.svd },
    { label: 'polar', present: (p) => !!p?.matrixDecompositions?.primaryDecompositions?.polar },
    { label: 'hessenberg', present: (p) => !!p?.matrixDecompositions?.similarityAndSpectral?.hessenbergDecomposition },
    { label: 'schur', present: (p) => !!p?.matrixDecompositions?.similarityAndSpectral?.schurDecomposition },
    { label: 'diagonalization', present: (p) => !!p?.matrixDecompositions?.similarityAndSpectral?.diagonalization },
    { label: 'symmetric spectral', present: (p) => !!p?.matrixDecompositions?.similarityAndSpectral?.symmetricSpectral },
    { label: 'bidiagonalization', present: (p) => !!p?.matrixDecompositions?.similarityAndSpectral?.bidiagonalization },
    { label: 'row space basis', present: (p) => !!p?.matrixDecompositions?.subspaceBases?.rowSpaceBasis },
    { label: 'column space basis', present: (p) => !!p?.matrixDecompositions?.subspaceBases?.columnSpaceBasis },
    { label: 'null space basis', present: (p) => !!p?.matrixDecompositions?.subspaceBases?.nullSpaceBasis }
  ]

  function seededRandom(seed) {
    let s = seed % 2147483647
    if (s <= 0) s += 2147483646
    return () => {
      s = (s * 16807) % 2147483647
      return (s - 1) / 2147483646
    }
  }

  function buildRandomMatrix(n, seed) {
    const rand = seededRandom(seed)
    const matrix = new Array(n)
    for (let i = 0; i < n; i++) {
      const row = new Array(n)
      for (let j = 0; j < n; j++) row[j] = rand() - 0.5
      matrix[i] = row
    }
    return matrix
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q
    const base = Math.floor(pos)
    const rest = pos - base
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base])
    }
    return sorted[base]
  }

  function removeOutliers(values) {
    if (!values || values.length < 4) return values ? values.slice() : []
    const sorted = values.slice().sort((a, b) => a - b)
    const q1 = quantile(sorted, 0.25)
    const q3 = quantile(sorted, 0.75)
    const iqr = q3 - q1
    const lower = q1 - 1.5 * iqr
    const upper = q3 + 1.5 * iqr
    return sorted.filter(v => v >= lower && v <= upper)
  }

  function average(values) {
    if (!values || values.length === 0) return null
    const sum = values.reduce((acc, v) => acc + v, 0)
    return sum / values.length
  }

  function estimateFlopsPerIteration(n) {
    const qr = (2 / 3) * n * n * n
    const lu = (2 / 3) * n * n * n
    const svd = 4 * n * n * n
    const schur = (10 / 3) * n * n * n
    return qr + lu + svd + schur
  }

  function validateRequiredDiagnostics(payload) {
    const scalar = payload?.basicProperties?.scalarInvariants
    const determinant = scalar?.determinant
    const primary = payload?.matrixDecompositions?.primaryDecompositions
    const similarity = payload?.matrixDecompositions?.similarityAndSpectral
    const derived = payload?.matrixDecompositions?.derivedMatrices
    const subspaces = payload?.matrixDecompositions?.subspaceBases
    const hasAllPrimary = !!(primary?.qr && primary?.lu && primary?.cholesky && primary?.svd && primary?.polar)
    const hasAllSimilarity = !!(
      similarity?.hessenbergDecomposition
      && similarity?.schurDecomposition
      && similarity?.diagonalization
      && similarity?.symmetricSpectral
      && similarity?.bidiagonalization
    )
    const hasInverse = !!derived?.inverse
    const hasBasisCalcs = !!(subspaces?.rowSpaceBasis && subspaces?.columnSpaceBasis && subspaces?.nullSpaceBasis)
    return determinant !== undefined && determinant !== null && hasInverse && hasAllPrimary && hasAllSimilarity && hasBasisCalcs
  }

  // initialize max from existing diagnostics (e.g., backend reported gflops on load)
  useEffect(() => {
    const g = diagnostics?.cpu?.gflops != null ? Number(diagnostics.cpu.gflops) : null
    if (!maxGflops && g && g > 0) {
      setMaxGflops(g)
      setRecentGflops(g)
      setBarWidthPercent(100)
      setBarColorClass('bg-primary')
    }
  }, [diagnostics, maxGflops])

  function logBenchmarkParts(payload, size, iteration) {
    for (const part of BENCHMARK_PARTS) {
      const ok = part.present(payload)
      setBenchmarkProgress((prev) => ({ ...prev, phase: `size ${size} iter ${iteration}: ${part.label}` }))
      console.debug(`[Benchmark] size=${size} iter=${iteration} part=${part.label} status=${ok ? 'ok' : 'missing'}`)
    }
  }

  function formatFlopsFromGflops(g) {
    if (g === null || g === undefined) return { value: '—', unit: 'FLOPs' }
    // backend provides GFLOPs (g)
    const flops = g * 1e9
    const units = [
      { name: 'KFLOPs', scale: 1e3 },
      { name: 'MFLOPs', scale: 1e6 },
      { name: 'GFLOPs', scale: 1e9 },
      { name: 'TFLOPs', scale: 1e12 }
    ]
    // find largest unit where value >= 1
    for (let i = units.length - 1; i >= 0; i--) {
      const u = units[i]
      if (flops >= u.scale) {
        const val = flops / u.scale
        const formatted = val >= 100 ? Math.round(val) : Math.round(val * 10) / 10
        return { value: formatted.toString(), unit: u.name }
      }
    }
    // smaller than KFLOPs
    const smallVal = Math.round(flops * 10) / 10
    return { value: smallVal.toString(), unit: 'FLOPs' }
  }

  function itemClass(id) {
    return `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors group ${
      active === id ? 'bg-primary/10 text-primary' : 'hover:bg-primary/5 dark:hover:bg-primary/10 text-slate-600 dark:text-slate-300'
    }`
  }

  function subItemClass(id) {
    return `flex items-center gap-3 px-3 py-2 ml-4 rounded-lg transition-colors group ${
      active === id ? 'bg-primary/10 text-primary' : 'hover:bg-primary/5 dark:hover:bg-primary/10 text-slate-600 dark:text-slate-300'
    }`
  }

  // Runs diagnostics for 2^n sizes, n=2..9, 5 iterations each.
  // Performance metric is taken only from the 512x512 runs.
  async function runSystemBenchmark() {
    const totalIterations = BENCHMARK_ITERATIONS
    setBenchmarkProgress({ complete: 0, total: totalIterations, phase: `starting Diagnostic benchmark (${BENCHMARK_TEST} ${BENCHMARK_SIZES[0]}x${BENCHMARK_SIZES[0]})` })
    setBenchmarkRunning(true)

    console.groupCollapsed('[Benchmark] runSystemBenchmark')
    console.info('[Benchmark] source: Diagnostic backend endpoint')
    console.info('[Benchmark] size:', BENCHMARK_SIZES[0])
    console.info('[Benchmark] test:', BENCHMARK_TEST)
    console.info('[Benchmark] iterations:', BENCHMARK_ITERATIONS)
    console.info('[Benchmark] totalIterations:', totalIterations)

    try {
      // mark as queued locally so diagnostics reflect queue impact
      setDiagnostics((prev) => ({ ...(prev || {}), cpu: { ...(prev?.cpu || {}), queuedJobs: (Number(prev?.cpu?.queuedJobs) || 0) + 1 } }))
      setBenchmarkProgress((prev) => ({ ...prev, phase: `calling /api/benchmark/diagnostic?sizex=${BENCHMARK_SIZES[0]}&sizey=${BENCHMARK_SIZES[0]}&test=${BENCHMARK_TEST}` }))
      const res = await fetch(`/api/benchmark/diagnostic?sizex=${BENCHMARK_SIZES[0]}&sizey=${BENCHMARK_SIZES[0]}&test=${encodeURIComponent(BENCHMARK_TEST)}&iterations=${BENCHMARK_ITERATIONS}`)
      if (!res.ok) throw new Error('diagnostic-benchmark-failed')
      const json = await res.json()
      const rows = Array.isArray(json?.iterations) ? json.iterations : []
      rows.forEach((row, idx) => {
        const op = row?.operation || 'unknown'
        const iter = row?.iteration ?? '?'
        const n = row?.n ?? '?'
        setBenchmarkProgress((prev) => ({
          ...prev,
          complete: Math.min(BENCHMARK_ITERATIONS, Math.max(prev.complete, Number(iter) || 0)),
          phase: `n=${n} iter=${iter}: ${op.toLowerCase()}`
        }))
        console.debug(`[Benchmark] n=${n} iter=${iter} op=${op} ms=${row?.ms} flopsPerSec=${row?.flopsPerSec}`)
      })

      setBenchmarkProgress((prev) => ({ ...prev, complete: BENCHMARK_ITERATIONS, phase: `completed Diagnostic benchmark (${BENCHMARK_TEST} ${BENCHMARK_SIZES[0]}x${BENCHMARK_SIZES[0]})` }))
      // update diagnostics from backend benchmark
      setDiagnostics((prev) => ({ ...(prev || {}), ...(json || {}) }))
      // determine new measured GFLOPs if present
      const measuredG = Number(json?.cpu?.gflops ?? json?.cpu?.benchmark?.gflopsAvg)
      if (!Number.isNaN(measuredG) && measuredG > 0) {
        setRecentGflops(measuredG)
        setDiagnostics((prev) => ({ ...(prev || {}), cpu: { ...(prev?.cpu || {}), gflops: measuredG } }))
        // Update max / bar behavior
        setMaxGflops((prevMax) => {
          if (!prevMax || measuredG > prevMax) {
            // new record: set max and show full bar and supercharged color
            setBarWidthPercent(100)
            setBarColorClass('bg-violet-600')
            return measuredG
          }
          // not a new max, set width proportional
          const width = Math.round((measuredG / prevMax) * 100)
          setBarWidthPercent(width)
          // if width dropped below 50% -> degraded state and sickly lavender bar
          if (width < 50) {
            setBarColorClass('bg-violet-200')
            setDiagnostics((prev) => ({ ...(prev || {}), cpu: { ...(prev?.cpu || {}), state: 'degraded' } }))
          } else {
            setBarColorClass('bg-primary')
            // if previously degraded but now above 50%, restore to online if backend says online
            setDiagnostics((prev) => ({ ...(prev || {}), cpu: { ...(prev?.cpu || {}), state: prev?.cpu?.state === 'degraded' ? 'online' : prev?.cpu?.state } }))
          }
          return prevMax
        })
      }
      console.info('[Benchmark] diagnostics updated from backend benchmark', json?.cpu)
    } catch (e) {
      console.error('[Benchmark] failed', e)
      setDiagnostics({ status: 'SERVICE_INTERRUPTION', cpu: { name: 'CPU', gflops: null, state: 'offline', queuedJobs: 0 } })
    } finally {
      // decrement queued jobs and finalize
      setDiagnostics((prev) => ({ ...(prev || {}), cpu: { ...(prev?.cpu || {}), queuedJobs: Math.max(0, (Number(prev?.cpu?.queuedJobs) || 1) - 1) } }))
      console.groupEnd()
      setBenchmarkRunning(false)
      setBenchmarkProgress((prev) => ({ ...prev, phase: '' }))
    }
  }

  return (
    <aside className="w-64 border-r border-slate-200 dark:border-slate-700 hidden lg:flex flex-col bg-white dark:bg-slate-800 p-4 overflow-y-auto transition-colors duration-300">
      <div className="space-y-6">
        {/* Libraries Section */}
        <div>
          <h3 className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-4 px-3">Libraries</h3>
          <div className="space-y-1">
            {/* Compute - Expandable */}
            <div>
              <button 
                onClick={() => setComputeExpanded(!computeExpanded)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-primary/5 dark:hover:bg-primary/10 text-slate-600 dark:text-slate-300 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-[20px]">calculate</span>
                  <span className="text-sm font-semibold">Compute</span>
                </div>
                <span className={`material-symbols-outlined text-[18px] transition-transform ${computeExpanded ? 'rotate-180' : ''}`}>
                  expand_more
                </span>
              </button>
              
              {computeExpanded && (
                <div className="mt-1 space-y-1">
                  {showCurrentAnalysis && (
                    <a className={subItemClass('analysis')} href="#">
                      <span className="material-symbols-outlined text-[18px]">analytics</span>
                      <span className="text-sm font-medium">Current Analysis</span>
                    </a>
                  )}
                  <a className={subItemClass('home')} href="/">
                    <span className="material-symbols-outlined text-[18px]">dashboard</span>
                    <span className="text-sm font-medium">My Computations</span>
                  </a>
                </div>
              )}
            </div>

            {/* Favorites */}
            <a className={itemClass('favorites')} href="/favorites">
              <span className="material-symbols-outlined text-[20px]">star</span>
              <span className="text-sm font-medium">Favorites</span>
            </a>

            {/* History */}
            <a className={itemClass('history')} href="/history">
              <span className="material-symbols-outlined text-[20px]">history</span>
              <span className="text-sm font-medium">History</span>
            </a>
          </div>
        </div>

        {/* Recent Sessions Section */}
        <div>
          <h3 className="text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-4 px-3">Recent Sessions</h3>
          {sessions.length === 0 ? (
            <div className="text-xs text-slate-400 dark:text-slate-500 px-3">(no recent sessions)</div>
          ) : (
            <div className="space-y-2">
              {sessions.map((s, i) => (
                <div 
                  key={i} 
                  className="group cursor-pointer px-3 py-2 rounded-lg hover:bg-primary/5 dark:hover:bg-primary/10 transition-colors" 
                  title={s.title} 
                  onClick={() => { import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(s.title)}/basic`)) }}
                >
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-slate-400 dark:text-slate-500 text-[18px] group-hover:text-primary transition-colors">description</span>
                    <span className="text-sm font-medium text-slate-600 dark:text-slate-300 group-hover:text-primary transition-colors truncate">{s.title}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 dark:text-slate-500 pl-[26px] mt-0.5">{new Date(s.ts).toLocaleString()}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="mt-auto pt-4">
        <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600 transition-colors duration-300">
          <div className="flex items-center gap-2 mb-2">
            <span className="material-symbols-outlined text-primary text-[18px]">info</span>
            <span className="text-[11px] font-bold text-slate-700 dark:text-slate-200 uppercase">System Status</span>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed mb-4">
            CPU is <span className={`font-bold ${cpuState === 'online' ? 'text-emerald-600' : (cpuState === 'offline' ? 'text-rose-600' : 'text-amber-500')}`}>{cpuState || 'offline'}</span> with queue position <span className="font-bold">{queuedJobs}</span>.
          </p>
            <div className="space-y-2">
            {/* ONLINE */}
            <div className={`flex items-center gap-2 ${systemStatus !== 'ONLINE' ? 'opacity-30 grayscale' : ''}`}>
              <div className="relative flex size-2">
                {systemStatus === 'ONLINE' ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full size-2 bg-emerald-500"></span>
                  </>
                ) : (
                  <span className="size-2 rounded-full bg-emerald-400"></span>
                )}
              </div>
              <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">ONLINE</span>
            </div>

              {/* BUSY (moderate queue) */}
              <div className={`flex items-center gap-2 ${systemStatus !== 'BUSY' ? 'opacity-30 grayscale' : ''}`}>
                <div className={`size-2 rounded-full ${systemStatus === 'BUSY' ? 'bg-sky-500' : 'bg-sky-400'}`}></div>
                <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">BUSY</span>
              </div>

            {/* LARGE QUEUE */}
            <div className={`flex items-center gap-2 ${systemStatus !== 'LARGE_QUEUE' ? 'opacity-30 grayscale' : ''}`}>
              <div className={`size-2 rounded-full ${systemStatus === 'LARGE_QUEUE' ? 'bg-yellow-400' : 'bg-yellow-300'}`}></div>
              <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">LARGE QUEUE</span>
            </div>

            {/* SERVICE INTERRUPTION */}
            <div className={`flex items-center gap-2 ${systemStatus !== 'SERVICE_INTERRUPTION' ? 'opacity-30 grayscale' : ''}`}>
              <div className={`size-2 rounded-full ${systemStatus === 'SERVICE_INTERRUPTION' ? 'bg-red-500' : 'bg-red-400'}`}></div>
              <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">SERVICE INTERRUPTION</span>
            </div>
          </div>

          <div className="mt-4">
            <button
              onClick={() => setShowInfoModal(true)}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-primary hover:bg-primary-hover text-white rounded-md text-[11px] font-bold uppercase tracking-wider transition-all shadow-md shadow-primary/20"
            >
              <span className="material-symbols-outlined text-[16px]">info</span>
              More Information
            </button>
          </div>
        </div>

        {showInfoModal && (
          <div aria-modal="true" className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6" role="dialog">
            <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity" onClick={() => setShowInfoModal(false)}></div>
            <div className="relative w-full max-w-2xl transform overflow-hidden rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-2xl transition-all flex flex-col max-h-[90vh]">
              <div className="flex items-center justify-between bg-primary px-6 py-4 border-b border-primary-hover shrink-0">
                <h3 className="text-lg font-bold text-white flex items-center gap-2.5">
                  <span className="material-symbols-outlined">dns</span>
                  System Status &amp; Performance
                </h3>
                <button onClick={() => setShowInfoModal(false)} className="text-white/80 hover:text-white hover:bg-white/10 rounded-full p-1 transition-colors">
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
              <div className="p-6 md:p-8 space-y-8 overflow-y-auto text-slate-900 dark:text-slate-100">
                <section>
                  <div className="flex flex-col items-center justify-center">
                    <div className="flex items-start justify-center gap-8 md:gap-12 mb-6 w-full">
                      <div className={`flex flex-col items-center gap-3 ${systemStatus !== 'ONLINE' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="relative flex items-center justify-center size-14 rounded-full bg-emerald-50 border-2 border-emerald-100 shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                          <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-20 animate-ping"></span>
                          <span className="material-symbols-outlined text-3xl text-emerald-500">check_circle</span>
                        </div>
                        <span className="text-[11px] font-bold text-emerald-600 tracking-wider">ONLINE</span>
                      </div>
                        <div className={`flex flex-col items-center gap-3 ${systemStatus !== 'BUSY' ? 'opacity-30 grayscale' : ''}`}>
                          <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 dark:bg-slate-700 border-2 border-slate-100 dark:border-slate-600">
                            <span className="material-symbols-outlined text-3xl text-sky-500">autorenew</span>
                          </div>
                          <span className="text-[11px] font-bold text-slate-600 dark:text-slate-300 tracking-wider">BUSY</span>
                        </div>
                      <div className={`flex flex-col items-center gap-3 ${systemStatus !== 'LARGE_QUEUE' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 dark:bg-slate-700 border-2 border-slate-100 dark:border-slate-600">
                          <span className="material-symbols-outlined text-3xl text-yellow-500">hourglass_top</span>
                        </div>
                        <span className="text-[11px] font-bold text-slate-600 dark:text-slate-300 tracking-wider">LARGE QUEUE</span>
                      </div>
                      <div className={`flex flex-col items-center gap-3 ${systemStatus !== 'SERVICE_INTERRUPTION' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 dark:bg-slate-700 border-2 border-slate-100 dark:border-slate-600">
                          <span className="material-symbols-outlined text-3xl text-red-500">warning</span>
                        </div>
                        <span className="text-[11px] font-bold text-slate-600 dark:text-slate-300 tracking-wider">SERVICE INTERRUPTION</span>
                      </div>
                    </div>
                      <div className={cpuPillClass}>
                        <span className="relative flex h-2 w-2">
                          <span className={cpuPingClass}></span>
                          <span className={cpuDotClass}></span>
                        </span>
                        {cpuState ? (
                          cpuState === 'online' ? `CPU Kernel is operational (queue=${queuedJobs})` : `CPU Kernel: ${cpuState}`
                        ) : 'CPU Kernel status'}
                      </div>
                  </div>
                </section>

                <section>
                  <h4 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest mb-4">Computation Nodes</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="relative group cursor-pointer">
                      <label className="block p-5 rounded-xl border-2 border-border-color dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-primary/60 transition-all shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                          <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
                            <span className="material-symbols-outlined text-2xl">memory</span>
                          </div>
                        </div>
                        <div>
                          <h5 className="text-sm font-bold text-slate-900 dark:text-slate-100">CPU Node</h5>
                          <p className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-1">{diagnostics && diagnostics.cpu && diagnostics.cpu.name ? diagnostics.cpu.name : 'CPU'}</p>
                        </div>
                        <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-700">
                          <div className="flex items-baseline gap-1">
                            {(() => {
                              let g = null
                              if (diagnostics && diagnostics.cpu) {
                                g = diagnostics.cpu.gflops != null ? diagnostics.cpu.gflops : (diagnostics.cpu.benchmark && diagnostics.cpu.benchmark.gflopsAvg ? diagnostics.cpu.benchmark.gflopsAvg : null)
                              }
                              const f = g != null ? formatFlopsFromGflops(g) : { value: '—', unit: 'FLOPs' }
                              return (
                                <>
                                  <span className="text-2xl font-mono font-bold text-slate-900 dark:text-slate-100">{f.value}</span>
                                  <span className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase">{f.unit}</span>
                                </>
                              )
                            })()}
                          </div>
                          <div className="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full mt-2 overflow-hidden">
                            <div className={`h-full ${barColorClass} rounded-full`} style={{ width: `${barWidthPercent}%` }}></div>
                          </div>
                        </div>
                      </label>
                    </div>

                    <div className="relative group cursor-pointer opacity-30">
                      <label className="block p-5 rounded-xl border-2 border-border-color dark:border-slate-600 bg-white dark:bg-slate-800 hover:border-primary/60 transition-all shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                          <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors">
                            <span className="material-symbols-outlined text-2xl">grid_view</span>
                          </div>
                        </div>
                        <div>
                          <h5 className="text-sm font-bold text-slate-900 dark:text-slate-100">GPU Node</h5>
                          <p className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-1">(disabled)</p>
                        </div>
                        <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-700">
                          <div className="flex items-baseline gap-1">
                            <span className="text-2xl font-mono font-bold text-slate-900">—</span>
                            <span className="text-xs font-bold text-slate-400 uppercase">TFLOPs</span>
                          </div>
                          <div className="w-full bg-slate-100 dark:bg-slate-700 h-1.5 rounded-full mt-2 overflow-hidden">
                            <div className="bg-indigo-500 h-full w-[0%] rounded-full"></div>
                          </div>
                        </div>
                      </label>
                    </div>
                  </div>
                </section>
              </div>
              <div className="p-6 md:px-8 md:pb-8 pt-0 bg-white dark:bg-slate-800 shrink-0">
                <button
                  onClick={() => { if (!benchmarkRunning) runSystemBenchmark() }}
                  disabled={benchmarkRunning}
                  className={`w-full group relative flex items-center justify-center gap-3 text-white text-base font-bold py-4 rounded-xl shadow-lg transition-all overflow-hidden ${benchmarkRunning ? 'bg-emerald-600' : 'bg-primary hover:bg-primary-hover shadow-primary/25'}`}
                >
                  <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></span>
                  {benchmarkRunning && (
                    <span className="absolute bottom-0 left-0 h-1 w-full bg-white/20">
                      <span
                        className="block h-full bg-emerald-400 transition-all"
                        style={{ width: `${benchmarkProgress.total ? Math.round((benchmarkProgress.complete / benchmarkProgress.total) * 100) : 0}%` }}
                      ></span>
                    </span>
                  )}
                  <span className="material-symbols-outlined">speed</span>
                  {benchmarkRunning
                    ? `Benchmarking...`
                    : 'Run System Benchmark'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
