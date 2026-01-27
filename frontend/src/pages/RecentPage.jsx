import React from 'react'
import MatrixHeader from '../components/MatrixHeader'
import MatrixSidebar from '../components/MatrixSidebar'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

/**
 * Recent queries view for previously submitted computations.
 */
export default function RecentPage() {
  const handleCompute = useMatrixCompute()

  return (
    <div className="bg-background-light font-display text-slate-900 min-h-screen">
      <MatrixHeader inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <MatrixSidebar active="recent" />
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
                  <tr className="hover:bg-slate-50/50 transition-colors">
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-3">
                        <div className="size-8 rounded bg-primary-light flex items-center justify-center text-primary">
                          <span className="material-symbols-outlined text-[18px]">psychology</span>
                        </div>
                        <span className="text-sm font-bold text-slate-800">eigenvalues of A</span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 border border-slate-200 rounded p-1 bg-white">
                          <div className="matrix-thumb-grid h-full w-full opacity-60">
                            <div className="bg-primary/20 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-primary/20 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-slate-100 rounded-sm"></div>
                            <div className="bg-primary/20 rounded-sm"></div>
                          </div>
                        </div>
                        <span className="text-xs font-mono text-slate-500">3x3 Real</span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <span className="text-xs font-medium text-slate-500">Oct 24, 2023 - 14:32</span>
                    </td>
                    <td className="px-6 py-5 text-right">
                      <button className="inline-flex items-center gap-2 px-4 py-1.5 rounded-lg border border-primary/20 text-primary text-xs font-bold hover:bg-primary hover:text-white transition-all">
                        <span className="material-symbols-outlined text-[16px]">play_arrow</span>
                        Re-run
                      </button>
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50/50 transition-colors">
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-3">
                        <div className="size-8 rounded bg-primary-light flex items-center justify-center text-primary">
                          <span className="material-symbols-outlined text-[18px]">rebase_edit</span>
                        </div>
                        <span className="text-sm font-bold text-slate-800">inverse of Hilbert 3x3</span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 border border-slate-200 rounded p-1 bg-white">
                          <div className="matrix-thumb-grid h-full w-full opacity-60">
                            <div className="bg-primary/40 rounded-sm"></div>
                            <div className="bg-primary/30 rounded-sm"></div>
                            <div className="bg-primary/20 rounded-sm"></div>
                            <div className="bg-primary/30 rounded-sm"></div>
                            <div className="bg-primary/20 rounded-sm"></div>
                            <div className="bg-primary/10 rounded-sm"></div>
                            <div className="bg-primary/20 rounded-sm"></div>
                            <div className="bg-primary/10 rounded-sm"></div>
                            <div className="bg-primary/5 rounded-sm"></div>
                          </div>
                        </div>
                        <span className="text-xs font-mono text-slate-500">3x3 Hilbert</span>
                      </div>
                    </td>
                    <td className="px-6 py-5">
                      <span className="text-xs font-medium text-slate-500">Oct 23, 2023 - 09:15</span>
                    </td>
                    <td className="px-6 py-5 text-right">
                      <button className="inline-flex items-center gap-2 px-4 py-1.5 rounded-lg border border-primary/20 text-primary text-xs font-bold hover:bg-primary hover:text-white transition-all">
                        <span className="material-symbols-outlined text-[16px]">play_arrow</span>
                        Re-run
                      </button>
                    </td>
                  </tr>
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

