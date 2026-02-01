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
    load()
    window.addEventListener('storage', load)
    return () => window.removeEventListener('storage', load)
  }, [])

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
          <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed mb-3">All computation kernels are currently operational.</p>
          <div className="flex items-center gap-1.5 text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
            <div className="size-2 rounded-full bg-emerald-500"></div>
            <span>UPTIME 99.9%</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
