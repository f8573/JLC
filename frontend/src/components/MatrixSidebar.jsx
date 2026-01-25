import React from 'react'

export default function MatrixSidebar({ active = 'analysis' }) {
  function itemClass(id) {
    return `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors group ${
      active === id ? 'bg-primary/10 text-primary' : 'hover:bg-primary/5 text-slate-600'
    }`
  }

  return (
    <aside className="w-64 border-r border-slate-200 hidden lg:flex flex-col bg-white p-4 overflow-y-auto">
      <div className="space-y-8">
        <div>
          <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4 px-3">Library</h3>
          <div className="space-y-1">
            <a className={itemClass('analysis')} href="#">
              <span className="material-symbols-outlined text-[20px]">analytics</span>
              <span className="text-sm font-semibold">Current Analysis</span>
            </a>
            <a className={itemClass('favorites')} href="/favorites">
              <span className="material-symbols-outlined text-[20px]">star</span>
              <span className="text-sm font-medium">Favorites</span>
            </a>
            <a className={itemClass('recent')} href="/recent">
              <span className="material-symbols-outlined text-[20px]">history</span>
              <span className="text-sm font-medium">Recent Queries</span>
            </a>
          </div>
        </div>
      </div>
    </aside>
  )
}
