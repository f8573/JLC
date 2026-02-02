import React, { useEffect, useState } from 'react'

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
  const [showInfoModal, setShowInfoModal] = useState(false)

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

  // Fetch diagnostics from backend. Backend expected to run benchmark on startup.
  async function fetchDiagnostics() {
    setRunning(true)
    try {
      // POST a small sample identity matrix so backend returns diagnostics
      const sample = { matrix: [[1,0,0],[0,1,0],[0,0,1]] }
      const res = await fetch('/api/diagnostics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sample)
      })
      if (!res.ok) throw new Error('no-diagnostics')
      const json = await res.json()
      setDiagnostics(json)
    } catch (e) {
      setDiagnostics({ status: 'SERVICE_INTERRUPTION', cpu: { name: 'CPU', gflops: null, state: 'offline' } })
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
        // debug incoming SSE payload
        console.debug('[SSE] status event received:', data)
        // only update when payload differs to avoid rerenders
        const current = diagnostics
        const a = JSON.stringify(current)
        const b = JSON.stringify(data)
        if (a !== b) setDiagnostics(data)
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

  const cpuState = diagnostics && diagnostics.cpu && diagnostics.cpu.state ? String(diagnostics.cpu.state).toLowerCase() : null
  const cpuColor = cpuState === 'online' ? 'emerald' : (cpuState === 'offline' ? 'rose' : 'amber')

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
            CPU is <span className={`font-bold ${diagnostics && diagnostics.cpu ? (diagnostics.cpu.state === 'online' ? 'text-emerald-600' : (diagnostics.cpu.state === 'offline' ? 'text-rose-600' : 'text-amber-500')) : 'text-rose-600'}`}>{diagnostics && diagnostics.cpu ? diagnostics.cpu.state : 'offline'}</span>.
          </p>
            <div className="space-y-2">
            {/* ONLINE */}
            <div className={`flex items-center gap-2 ${diagnostics && diagnostics.status !== 'ONLINE' ? 'opacity-30 grayscale' : ''}`}>
              <div className="relative flex size-2">
                {diagnostics && diagnostics.status === 'ONLINE' ? (
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
              <div className={`flex items-center gap-2 ${diagnostics && diagnostics.status !== 'BUSY' ? 'opacity-30 grayscale' : ''}`}>
                <div className={`size-2 rounded-full ${diagnostics && diagnostics.status === 'BUSY' ? 'bg-sky-500' : 'bg-sky-400'}`}></div>
                <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">BUSY</span>
              </div>

            {/* LARGE QUEUE */}
            <div className={`flex items-center gap-2 ${diagnostics && diagnostics.status !== 'LARGE_QUEUE' ? 'opacity-30 grayscale' : ''}`}>
              <div className={`size-2 rounded-full ${diagnostics && diagnostics.status === 'LARGE_QUEUE' ? 'bg-yellow-400' : 'bg-yellow-300'}`}></div>
              <span className="text-[10px] font-bold text-slate-800 dark:text-slate-200">LARGE QUEUE</span>
            </div>

            {/* SERVICE INTERRUPTION */}
            <div className={`flex items-center gap-2 ${diagnostics && diagnostics.status !== 'SERVICE_INTERRUPTION' ? 'opacity-30 grayscale' : ''}`}>
              <div className={`size-2 rounded-full ${diagnostics && diagnostics.status === 'SERVICE_INTERRUPTION' ? 'bg-red-500' : 'bg-red-400'}`}></div>
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
            <div className="relative w-full max-w-2xl transform overflow-hidden rounded-xl bg-white shadow-2xl transition-all flex flex-col max-h-[90vh]">
              <div className="flex items-center justify-between bg-primary px-6 py-4 border-b border-primary-hover shrink-0">
                <h3 className="text-lg font-bold text-white flex items-center gap-2.5">
                  <span className="material-symbols-outlined">dns</span>
                  System Status &amp; Performance
                </h3>
                <button onClick={() => setShowInfoModal(false)} className="text-white/80 hover:text-white hover:bg-white/10 rounded-full p-1 transition-colors">
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
              <div className="p-6 md:p-8 space-y-8 overflow-y-auto">
                <section>
                  <div className="flex flex-col items-center justify-center">
                    <div className="flex items-start justify-center gap-8 md:gap-12 mb-6 w-full">
                      <div className={`flex flex-col items-center gap-3 ${diagnostics && diagnostics.status !== 'ONLINE' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="relative flex items-center justify-center size-14 rounded-full bg-emerald-50 border-2 border-emerald-100 shadow-[0_0_20px_rgba(16,185,129,0.3)]">
                          <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-20 animate-ping"></span>
                          <span className="material-symbols-outlined text-3xl text-emerald-500">check_circle</span>
                        </div>
                        <span className="text-[11px] font-bold text-emerald-600 tracking-wider">ONLINE</span>
                      </div>
                        <div className={`flex flex-col items-center gap-3 ${diagnostics && diagnostics.status !== 'BUSY' ? 'opacity-30 grayscale' : ''}`}>
                          <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 border-2 border-slate-100">
                            <span className="material-symbols-outlined text-3xl text-sky-500">autorenew</span>
                          </div>
                          <span className="text-[11px] font-bold text-slate-600 tracking-wider">BUSY</span>
                        </div>
                      <div className={`flex flex-col items-center gap-3 ${diagnostics && diagnostics.status !== 'LARGE_QUEUE' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 border-2 border-slate-100">
                          <span className="material-symbols-outlined text-3xl text-yellow-500">hourglass_top</span>
                        </div>
                        <span className="text-[11px] font-bold text-slate-600 tracking-wider">LARGE QUEUE</span>
                      </div>
                      <div className={`flex flex-col items-center gap-3 ${diagnostics && diagnostics.status !== 'SERVICE_INTERRUPTION' ? 'opacity-30 grayscale' : ''}`}>
                        <div className="flex items-center justify-center size-14 rounded-full bg-slate-50 border-2 border-slate-100">
                          <span className="material-symbols-outlined text-3xl text-red-500">warning</span>
                        </div>
                        <span className="text-[11px] font-bold text-slate-600 tracking-wider">SERVICE INTERRUPTION</span>
                      </div>
                    </div>
                      <div className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-${cpuColor}-50 border border-${cpuColor}-100 text-${cpuColor}-700 text-sm font-medium`}>
                        <span className="relative flex h-2 w-2">
                          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full bg-${cpuColor}-400 opacity-75`}></span>
                          <span className={`relative inline-flex rounded-full h-2 w-2 bg-${cpuColor}-500`}></span>
                        </span>
                        {cpuState ? (
                          cpuState === 'online' ? 'CPU Kernel is operational' : `CPU Kernel: ${diagnostics.cpu.state.replace(/_/g, ' ')}`
                        ) : 'CPU Kernel status'}
                      </div>
                  </div>
                </section>

                <section>
                  <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Computation Nodes</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="relative group cursor-pointer">
                      <label className="block p-5 rounded-xl border-2 border-border-color bg-white hover:border-primary/60 transition-all shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                          <div className="p-2.5 rounded-lg bg-slate-100 text-slate-600 transition-colors">
                            <span className="material-symbols-outlined text-2xl">memory</span>
                          </div>
                        </div>
                        <div>
                          <h5 className="text-sm font-bold text-slate-900">CPU Node</h5>
                          <p className="text-xs text-slate-500 font-mono mt-1">{diagnostics && diagnostics.cpu && diagnostics.cpu.name ? diagnostics.cpu.name : 'CPU'}</p>
                        </div>
                        <div className="mt-4 pt-4 border-t border-slate-100">
                          <div className="flex items-baseline gap-1">
                            {(() => {
                              let g = null
                              if (diagnostics && diagnostics.cpu) {
                                g = diagnostics.cpu.gflops != null ? diagnostics.cpu.gflops : (diagnostics.cpu.benchmark && diagnostics.cpu.benchmark.gflopsAvg ? diagnostics.cpu.benchmark.gflopsAvg : null)
                              }
                              const f = g != null ? formatFlopsFromGflops(g) : { value: '—', unit: 'FLOPs' }
                              return (
                                <>
                                  <span className="text-2xl font-mono font-bold text-slate-900">{f.value}</span>
                                  <span className="text-xs font-bold text-slate-400 uppercase">{f.unit}</span>
                                </>
                              )
                            })()}
                          </div>
                          <div className="w-full bg-slate-100 h-1.5 rounded-full mt-2 overflow-hidden">
                            <div className="bg-primary h-full w-[45%] rounded-full"></div>
                          </div>
                        </div>
                      </label>
                    </div>

                    <div className="relative group cursor-pointer opacity-30">
                      <label className="block p-5 rounded-xl border-2 border-border-color bg-white hover:border-primary/60 transition-all shadow-sm">
                        <div className="flex justify-between items-start mb-4">
                          <div className="p-2.5 rounded-lg bg-slate-100 text-slate-600 transition-colors">
                            <span className="material-symbols-outlined text-2xl">grid_view</span>
                          </div>
                        </div>
                        <div>
                          <h5 className="text-sm font-bold text-slate-900">GPU Node</h5>
                          <p className="text-xs text-slate-500 font-mono mt-1">(disabled)</p>
                        </div>
                        <div className="mt-4 pt-4 border-t border-slate-100">
                          <div className="flex items-baseline gap-1">
                            <span className="text-2xl font-mono font-bold text-slate-900">—</span>
                            <span className="text-xs font-bold text-slate-400 uppercase">TFLOPs</span>
                          </div>
                          <div className="w-full bg-slate-100 h-1.5 rounded-full mt-2 overflow-hidden">
                            <div className="bg-indigo-500 h-full w-[0%] rounded-full"></div>
                          </div>
                        </div>
                      </label>
                    </div>
                  </div>
                </section>
              </div>
              <div className="p-6 md:px-8 md:pb-8 pt-0 bg-white shrink-0">
                <button onClick={() => { fetchDiagnostics(); setShowInfoModal(false); }} className="w-full group relative flex items-center justify-center gap-3 bg-primary hover:bg-primary-hover text-white text-base font-bold py-4 rounded-xl shadow-lg shadow-primary/25 transition-all overflow-hidden">
                  <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></span>
                  <span className="material-symbols-outlined">speed</span>
                  Run System Benchmark
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
