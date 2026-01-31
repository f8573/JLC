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

  React.useEffect(() => {
    try {
      const raw = localStorage.getItem('recentSessions')
      const arr = raw ? JSON.parse(raw) : []
      setRecent(arr)
    } catch (e) {
      setRecent([])
    }
  }, [])

  return (
    <div className="bg-background-light font-display text-slate-900 h-screen overflow-hidden">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="recent" />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light">
          <div className="max-w-[1200px] mx-auto p-8 space-y-6">
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
                <a className="hover:text-primary" href="#">Library</a>
                <span className="material-symbols-outlined text-[14px]">chevron_right</span>
                <span className="text-slate-900">Recent Queries</span>
              </div>
              <div className="flex flex-wrap justify-between items-end gap-4 mt-2">
                <div className="space-y-1">
                  <h1 className="text-3xl font-black tracking-tight text-slate-900 uppercase">Recent Queries</h1>
                  <p className="text-slate-500 text-sm">Review and re-run your previous matrix computations</p>
                </div>
                <div className="flex gap-3">
                  <button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700">
                    <span className="material-symbols-outlined text-[20px]">delete</span>
                    Clear History
                  </button>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 uppercase tracking-widest">Query</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 uppercase tracking-widest">Matrix Input</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 uppercase tracking-widest">Date</th>
                    <th className="px-6 py-4 text-[11px] font-bold text-slate-400 uppercase tracking-widest text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {recent.length === 0 && (
                    <tr>
                      <td colSpan={4} className="px-6 py-8 text-center text-slate-400">No recent sessions</td>
                    </tr>
                  )}
                  {recent.map((item, idx) => (
                    <tr key={item.ts || idx} className="hover:bg-slate-50/50 transition-colors cursor-pointer" onClick={() => { import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(item.title)}/basic`)) }}>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-3">
                          <div className="size-8 rounded bg-primary-light flex items-center justify-center text-primary">
                            <span className="material-symbols-outlined text-[18px]">history</span>
                          </div>
                          <span className="text-sm font-bold text-slate-800 truncate">{(item.title || '').slice(0, 60)}</span>
                        </div>
                      </td>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 border border-slate-200 rounded p-1 bg-white">
                            <div className="matrix-thumb-grid h-full w-full opacity-60" />
                          </div>
                          <span className="text-xs font-mono text-slate-500">{item.rows && item.cols ? `${item.rows}x${item.cols}` : 'Matrix'}</span>
                        </div>
                      </td>
                      <td className="px-6 py-5">
                        <span className="text-xs font-medium text-slate-500">{item.ts ? new Date(item.ts).toLocaleString() : '—'}</span>
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
              <div className="p-6 bg-slate-50/50 border-t border-slate-200 flex items-center justify-between">
                <span className="text-xs text-slate-500">Showing 4 of 128 queries</span>
                <div className="flex gap-2">
                  <button className="p-2 border border-slate-200 rounded bg-white hover:bg-slate-50 text-slate-400">
                    <span className="material-symbols-outlined text-[18px]">chevron_left</span>
                  </button>
                  <button className="p-2 border border-slate-200 rounded bg-white hover:bg-slate-50 text-slate-400">
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

