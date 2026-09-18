import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

/**
 * Recent queries view for previously submitted computations.
 */
export default function RecentPage() {
  const handleCompute = useMatrixCompute()
  const [recent, setRecent] = React.useState([])

  const loadRecent = () => {
    try {
      const raw = localStorage.getItem('recentSessions')
      const arr = raw ? JSON.parse(raw) : []
      setRecent(arr)
    } catch (e) {
      setRecent([])
    }
  }

  React.useEffect(() => {
    loadRecent()
    
    // Listen for history-cleared events from settings
    const handleHistoryCleared = () => {
      setRecent([])
    }
    window.addEventListener('history-cleared', handleHistoryCleared)
    return () => window.removeEventListener('history-cleared', handleHistoryCleared)
  }, [])

  return (
    <div className="bg-background-light dark:bg-slate-900 font-display text-slate-900 dark:text-slate-100 h-screen overflow-hidden transition-colors duration-300">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="recent" />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light dark:bg-slate-900 transition-colors duration-300">
          <div className="max-w-[1200px] mx-auto p-8 space-y-6">
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-400 dark:text-slate-500">
                <a className="hover:text-primary" href="#">Library</a>
                <span className="material-symbols-outlined text-[14px]">chevron_right</span>
                <span className="text-slate-900 dark:text-white">Recent Queries</span>
              </div>
              <div className="flex flex-wrap justify-between items-end gap-4 mt-2">
                <div className="space-y-1">
                  <h1 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white uppercase">Recent Queries</h1>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">Review and re-run your previous matrix computations</p>
                </div>
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-5 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 rounded-xl text-sm font-bold transition-all text-slate-700 dark:text-slate-200">
                    <span className="material-symbols-outlined text-[20px]">delete</span>
                    Clear History
                  </button>
                </div>
              </div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden transition-colors duration-300">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-700/50 border-b border-slate-200 dark:border-slate-700">
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Query</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Matrix Input</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Date</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                  {recent.length === 0 && (
                    <tr>
                      <td colSpan={4} className="px-6 py-8 text-center text-slate-400 dark:text-slate-500">No recent sessions</td>
                    </tr>
                  )}
                  {recent.map((item, idx) => (
                    <tr key={item.ts || idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-700/50 transition-colors cursor-pointer" onClick={() => { import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(item.title)}/basic`)) }}>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-3">
                          <div className="size-8 rounded bg-primary-light dark:bg-primary/20 flex items-center justify-center text-primary">
                            <span className="material-symbols-outlined text-[18px]">history</span>
                          </div>
                          <span className="text-sm font-bold text-slate-800 dark:text-slate-200 truncate">{(item.title || '').slice(0, 60)}</span>
                        </div>
                      </td>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 border border-slate-200 dark:border-slate-600 rounded p-1 bg-white dark:bg-slate-700">
                            <div className="matrix-thumb-grid h-full w-full opacity-60" />
                          </div>
                          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">{item.rows && item.cols ? `${item.rows}x${item.cols}` : 'Matrix'}</span>
                        </div>
                      </td>
                      <td className="px-6 py-5">
                        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">{item.ts ? new Date(item.ts).toLocaleString() : '—'}</span>
                      </td>
                      <td className="px-6 py-5 text-right">
                        <button onClick={(e) => { e.stopPropagation(); import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(item.title)}/basic`)) }} className="inline-flex items-center gap-2 px-4 py-1.5 rounded-lg border border-primary/20 text-primary text-xs font-bold hover:bg-primary hover:text-white transition-all">
                          <span className="material-symbols-outlined text-[16px]">play_arrow</span>
                          Re-run
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="p-6 bg-slate-50/50 dark:bg-slate-700/30 border-t border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <span className="text-xs text-slate-500 dark:text-slate-400">Showing 4 of 128 queries</span>
                <div className="flex gap-2">
                  <button className="p-2 border border-slate-200 dark:border-slate-600 rounded bg-white dark:bg-slate-700 hover:bg-slate-50 dark:hover:bg-slate-600 text-slate-400 dark:text-slate-500">
                    <span className="material-symbols-outlined text-[18px]">chevron_left</span>
                  </button>
                  <button className="p-2 border border-slate-200 dark:border-slate-600 rounded bg-white dark:bg-slate-700 hover:bg-slate-50 dark:hover:bg-slate-600 text-slate-400 dark:text-slate-500">
                    <span className="material-symbols-outlined text-[18px]">chevron_right</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

