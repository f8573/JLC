import React, { useEffect, useState } from 'react'

export default function Sidebar({ active = 'home' }) {
  const [sessions, setSessions] = useState([])

  useEffect(() => {
    function load() {
      try {
        const raw = localStorage.getItem('recentSessions')
        const arr = raw ? JSON.parse(raw) : []
        setSessions(arr)
      } catch (e) {
        setSessions([])
      }
    }
    load()
    window.addEventListener('storage', load)
    return () => window.removeEventListener('storage', load)
  }, [])

  return (
    <aside className="w-64 border-r border-border-color bg-white flex flex-col shrink-0">
      <div className="p-4 border-b border-border-color">
        <h3 className="text-[11px] font-bold text-slate-gray uppercase tracking-widest mb-4">Library</h3>
        <div className="space-y-1">
          <a
            href="/"
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium group transition-all ${
              active === 'home'
                ? 'bg-purple-light text-primary'
                : 'text-slate-600 hover:text-primary hover:bg-purple-light'
            }`}
          >
            <span className={`material-symbols-outlined text-[20px] ${active === 'home' ? 'text-primary' : 'group-hover:text-primary'}`}>dashboard</span>
            <span>My Computations</span>
          </a>
          <a
            href="/favorites"
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium group transition-all ${
              active === 'favorites'
                ? 'bg-purple-light text-primary'
                : 'text-slate-600 hover:text-primary hover:bg-purple-light'
            }`}
          >
            <span className={`material-symbols-outlined text-[20px] ${active === 'favorites' ? 'text-primary' : 'group-hover:text-primary'}`}>star</span>
            <span>Favorites</span>
          </a>
          <a
            href="/history"
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium group transition-all ${
              active === 'history'
                ? 'bg-purple-light text-primary'
                : 'text-slate-600 hover:text-primary hover:bg-purple-light'
            }`}
          >
            <span className={`material-symbols-outlined text-[20px] ${active === 'history' ? 'text-primary' : 'group-hover:text-primary'}`}>history</span>
            <span>History</span>
          </a>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-[11px] font-bold text-slate-gray uppercase tracking-widest mb-4">Recent Sessions</h3>
        {sessions.length === 0 ? (
          <div className="text-[11px] text-slate-400">(no recent sessions)</div>
        ) : (
          <div className="space-y-3">
            {sessions.map((s, i) => (
              <div key={i} className="group cursor-pointer" title={s.title}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="material-symbols-outlined text-slate-400 text-[16px]">description</span>
                  <span className="text-xs font-semibold text-slate-700 group-hover:text-primary transition-colors">{s.title}</span>
                </div>
                <p className="text-[10px] text-slate-500 pl-6 leading-relaxed">{new Date(s.ts).toLocaleString()}</p>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="p-4 bg-slate-50">
        <div className="p-4 rounded-lg bg-white border border-border-color shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <span className="material-symbols-outlined text-primary text-sm">info</span>
            <span className="text-[11px] font-bold text-slate-900 uppercase">System Status</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed mb-3">All computation kernels are currently operational.</p>
          <div className="flex items-center gap-1.5 text-[10px] text-emerald-600 font-bold">
            <div className="size-1.5 rounded-full bg-emerald-500"></div>
            <span>UPTIME 99.9%</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
