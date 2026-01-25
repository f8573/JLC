import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'

export default function FavoritesPage() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#ffffff] text-slate-800 font-sans selection:bg-primary/20">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active="favorites" />
        <main className="flex-1 overflow-y-auto bg-white math-grid relative flex flex-col">
          <div className="p-8 max-w-7xl mx-auto w-full">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
              <div>
                <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Favorites</h1>
                <p className="text-sm text-slate-500">Access and manage your starred matrices and computations.</p>
              </div>
              <div className="w-full md:w-80">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                    <span className="material-symbols-outlined text-[20px]">filter_list</span>
                  </div>
                  <input
                    className="block w-full bg-white border border-border-color rounded-lg py-2 pl-10 pr-3 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all shadow-sm"
                    placeholder="Search favorites..."
                    type="text"
                  />
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              <div className="bg-white border border-border-color rounded-xl p-5 shadow-sm hover:shadow-md hover:border-primary/30 transition-all group cursor-pointer relative">
                <button className="absolute top-4 right-4 text-slate-300 hover:text-red-500 transition-colors">
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
                <div className="flex justify-center mb-6">
                  <div className="matrix-preview-bracket px-4 py-3 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-3 gap-2 text-[10px] font-mono text-slate-600">
                      <span>1</span> <span>0</span> <span>0</span>
                      <span>0</span> <span>1</span> <span>0</span>
                      <span>0</span> <span>0</span> <span>1</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="material-symbols-outlined text-primary text-[16px] fill-1">star</span>
                    <h3 className="font-bold text-slate-900 text-sm truncate">Identity Matrix - Standard</h3>
                  </div>
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                    <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-tighter">3 x 3</span>
                    <span className="text-[10px] text-slate-400 italic">Added Oct 24, 2023</span>
                  </div>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-5 shadow-sm hover:shadow-md hover:border-primary/30 transition-all group cursor-pointer relative">
                <button className="absolute top-4 right-4 text-slate-300 hover:text-red-500 transition-colors">
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
                <div className="flex justify-center mb-6">
                  <div className="matrix-preview-bracket px-4 py-3 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-2 gap-3 text-[10px] font-mono text-slate-600">
                      <span>0.6</span> <span>-0.8</span>
                      <span>0.8</span> <span>0.6</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="material-symbols-outlined text-primary text-[16px] fill-1">star</span>
                    <h3 className="font-bold text-slate-900 text-sm truncate">Orthogonal Matrix</h3>
                  </div>
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                    <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-tighter">2 x 2</span>
                    <span className="text-[10px] text-slate-400 italic">Added Nov 12, 2023</span>
                  </div>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-5 shadow-sm hover:shadow-md hover:border-primary/30 transition-all group cursor-pointer relative">
                <button className="absolute top-4 right-4 text-slate-300 hover:text-red-500 transition-colors">
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
                <div className="flex justify-center mb-6">
                  <div className="matrix-preview-bracket px-4 py-3 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-4 gap-2 text-[10px] font-mono text-slate-600">
                      <span>4</span> <span>2</span> <span>3</span> <span>1</span>
                      <span>1</span> <span>5</span> <span>7</span> <span>2</span>
                      <span>0</span> <span>2</span> <span>6</span> <span>3</span>
                      <span>0</span> <span>0</span> <span>1</span> <span>4</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="material-symbols-outlined text-primary text-[16px] fill-1">star</span>
                    <h3 className="font-bold text-slate-900 text-sm truncate">Hessenberg Matrix</h3>
                  </div>
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                    <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-tighter">4 x 4</span>
                    <span className="text-[10px] text-slate-400 italic">Added Dec 02, 2023</span>
                  </div>
                </div>
              </div>
              <div className="bg-white border border-border-color rounded-xl p-5 shadow-sm hover:shadow-md hover:border-primary/30 transition-all group cursor-pointer relative">
                <button className="absolute top-4 right-4 text-slate-300 hover:text-red-500 transition-colors">
                  <span className="material-symbols-outlined text-[20px]">delete</span>
                </button>
                <div className="flex justify-center mb-6">
                  <div className="matrix-preview-bracket px-4 py-3 bg-slate-50 rounded-sm">
                    <div className="grid grid-cols-2 gap-3 text-[10px] font-mono text-slate-600">
                      <span>cos ?</span> <span>-sin ?</span>
                      <span>sin ?</span> <span>cos ?</span>
                    </div>
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="material-symbols-outlined text-primary text-[16px] fill-1">star</span>
                    <h3 className="font-bold text-slate-900 text-sm truncate">Rotation Matrix (Symbolic)</h3>
                  </div>
                  <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                    <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-tighter">2 x 2</span>
                    <span className="text-[10px] text-slate-400 italic">Added Jan 05, 2024</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-12 flex flex-col items-center justify-center py-16 px-4 bg-white/50 border-2 border-dashed border-border-color rounded-2xl">
              <span className="material-symbols-outlined text-slate-300 text-5xl mb-4">auto_awesome</span>
              <h3 className="text-slate-900 font-bold mb-1">Need to add more?</h3>
              <p className="text-slate-500 text-sm text-center max-w-xs mb-6">
                Star matrices in the Input Console to quickly access them here for future computations.
              </p>
              <a href="/" className="flex items-center gap-2 px-6 py-2 bg-primary text-white text-sm font-bold rounded-lg hover:bg-primary-hover transition-all">
                <span className="material-symbols-outlined text-lg">add_circle</span>
                Go to Editor
              </a>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

