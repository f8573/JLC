import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'

/**
 * History view listing completed matrix sessions and actions.
 */
export default function HistoryPage() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#ffffff] text-slate-800 font-sans selection:bg-primary/20">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active="history" />
        <main className="flex-1 overflow-y-auto bg-white math-grid relative flex flex-col">
          <div className="p-8 max-w-7xl mx-auto w-full">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
              <div>
                <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Computation History</h1>
                <p className="text-sm text-slate-500">Review and restore your past matrix operations.</p>
              </div>
              <div className="w-full md:w-80">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <span className="material-symbols-outlined text-[20px]">search</span>
                  </div>
                  <input
                    className="block w-full bg-white border border-border-color rounded-lg py-2 pl-10 pr-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all shadow-sm"
                    placeholder="Filter history..."
                    type="text"
                  />
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="bg-white border border-border-color rounded-xl p-4 flex items-center gap-6 hover:border-primary/30 transition-all group shadow-sm">
                <div className="w-32 flex justify-center shrink-0">
                  <div className="matrix-preview-bracket px-3 py-2 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono text-slate-600">
                      <span>1</span> <span>2</span>
                      <span>3</span> <span>4</span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-slate-900 truncate">Matrix Inversion</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Operation: <span className="text-primary font-medium">Computed Inverse</span></p>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-[11px] font-medium text-slate-400 block mb-1">Today, 2:45 PM</span>
                  <div className="flex items-center gap-2 justify-end">
                    <span className="text-[10px] font-mono font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">2x2</span>
                  </div>
                </div>
                <div className="pl-4 border-l border-slate-100 shrink-0">
                  <button className="flex items-center gap-1.5 px-4 py-1.5 text-primary bg-purple-light hover:bg-primary hover:text-white text-xs font-bold rounded-lg transition-all">
                    <span className="material-symbols-outlined text-[18px]">restore</span>
                    Restore
                  </button>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-4 flex items-center gap-6 hover:border-primary/30 transition-all group shadow-sm">
                <div className="w-32 flex justify-center shrink-0">
                  <div className="matrix-preview-bracket px-3 py-2 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-3 gap-x-2 gap-y-1 text-[10px] font-mono text-slate-600">
                      <span>1</span> <span>0</span> <span>0</span>
                      <span>0</span> <span>1</span> <span>0</span>
                      <span>0</span> <span>0</span> <span>1</span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-slate-900 truncate">Identity Analysis</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Operation: <span className="text-primary font-medium">Analyzed Properties</span></p>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-[11px] font-medium text-slate-400 block mb-1">Yesterday, 11:20 AM</span>
                  <div className="flex items-center gap-2 justify-end">
                    <span className="text-[10px] font-mono font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">3x3</span>
                  </div>
                </div>
                <div className="pl-4 border-l border-slate-100 shrink-0">
                  <button className="flex items-center gap-1.5 px-4 py-1.5 text-primary bg-purple-light hover:bg-primary hover:text-white text-xs font-bold rounded-lg transition-all">
                    <span className="material-symbols-outlined text-[18px]">restore</span>
                    Restore
                  </button>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-4 flex items-center gap-6 hover:border-primary/30 transition-all group shadow-sm">
                <div className="w-32 flex justify-center shrink-0">
                  <div className="matrix-preview-bracket px-3 py-2 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono text-slate-600">
                      <span>0.6</span> <span>-0.8</span>
                      <span>0.8</span> <span>0.6</span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-slate-900 truncate">Orthogonal Matrix</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Operation: <span className="text-primary font-medium">Transposed</span></p>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-[11px] font-medium text-slate-400 block mb-1">Oct 26, 2023, 9:15 AM</span>
                  <div className="flex items-center gap-2 justify-end">
                    <span className="text-[10px] font-mono font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">2x2</span>
                  </div>
                </div>
                <div className="pl-4 border-l border-slate-100 shrink-0">
                  <button className="flex items-center gap-1.5 px-4 py-1.5 text-primary bg-purple-light hover:bg-primary hover:text-white text-xs font-bold rounded-lg transition-all">
                    <span className="material-symbols-outlined text-[18px]">restore</span>
                    Restore
                  </button>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-4 flex items-center gap-6 hover:border-primary/30 transition-all group shadow-sm">
                <div className="w-32 flex justify-center shrink-0">
                  <div className="matrix-preview-bracket px-3 py-2 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-4 gap-x-1.5 gap-y-1 text-[9px] font-mono text-slate-600">
                      <span>4</span> <span>2</span> <span>3</span> <span>1</span>
                      <span>1</span> <span>5</span> <span>7</span> <span>2</span>
                      <span>0</span> <span>2</span> <span>6</span> <span>3</span>
                      <span>0</span> <span>0</span> <span>1</span> <span>4</span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-bold text-slate-900 truncate">Hessenberg Matrix</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Operation: <span className="text-primary font-medium">QR Decomposition</span></p>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-[11px] font-medium text-slate-400 block mb-1">Oct 25, 2023, 4:50 PM</span>
                  <div className="flex items-center gap-2 justify-end">
                    <span className="text-[10px] font-mono font-bold text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">4x4</span>
                  </div>
                </div>
                <div className="pl-4 border-l border-slate-100 shrink-0">
                  <button className="flex items-center gap-1.5 px-4 py-1.5 text-primary bg-purple-light hover:bg-primary hover:text-white text-xs font-bold rounded-lg transition-all">
                    <span className="material-symbols-outlined text-[18px]">restore</span>
                    Restore
                  </button>
                </div>
              </div>
            </div>
            <div className="mt-8 flex justify-center">
              <button className="text-sm font-semibold text-slate-500 hover:text-primary transition-colors flex items-center gap-2">
                <span>Load more sessions</span>
                <span className="material-symbols-outlined text-[20px]">expand_more</span>
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

