import React, { useEffect, useState } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { parseMatrixString } from '../utils/diagnostics'
import MatrixLatex from '../components/matrix/MatrixLatex'

function MatrixPreview({ matrixString }) {
  const matrix = parseMatrixString(matrixString) || []
  const rows = matrix.length
  const cols = matrix[0] ? matrix[0].length : 0
  const rmax = Math.min(4, rows)
  const cmax = Math.min(4, cols)
  if (rows === 0) return <div className="px-4 py-3 bg-slate-50 rounded-sm">Empty</div>
  const previewMatrix = matrix.slice(0, rmax).map(row => row.slice(0, cmax))
  return (
    <div className="px-3 py-2 bg-slate-50 rounded-sm">
      <MatrixLatex data={previewMatrix} className="text-[10px] math-font text-slate-600" />
    </div>
  )
}

/**
 * Favorites catalog for starred matrices and computations.
 */
export default function FavoritesPage() {
  const [favorites, setFavorites] = useState([])
  const [search, setSearch] = useState('')
  const [typeFilter, setTypeFilter] = useState('any')
  const [sparsityFilter, setSparsityFilter] = useState('any')
  const [dimFilter, setDimFilter] = useState('')
  const [editing, setEditing] = useState({})

  useEffect(() => {
    try {
      const raw = localStorage.getItem('favorites')
      const arr = raw ? JSON.parse(raw) : []
      setFavorites(arr)
    } catch (e) {
      setFavorites([])
    }
  }, [])

  function persist(arr) {
    localStorage.setItem('favorites', JSON.stringify(arr))
    setFavorites(arr)
  }

  function removeFavorite(idx) {
    const arr = favorites.slice()
    arr.splice(idx, 1)
    persist(arr)
  }

  function startEdit(idx) {
    setEditing({ idx, name: favorites[idx].name })
  }

  function saveEdit() {
    const arr = favorites.slice()
    arr[editing.idx] = { ...arr[editing.idx], name: editing.name }
    persist(arr)
    setEditing({})
  }

  function filtered() {
    return favorites.filter(f => {
      if (search && !(f.name || f.matrixString || '').toLowerCase().includes(search.toLowerCase())) return false
      if (typeFilter !== 'any' && (f.type || 'general') !== typeFilter) return false
      if (sparsityFilter !== 'any') {
        const d = f.density ?? 0
        if (sparsityFilter === 'sparse' && d > 0.2) return false
        if (sparsityFilter === 'medium' && (d <= 0.2 || d >= 0.6)) return false
        if (sparsityFilter === 'dense' && d < 0.6) return false
      }
      if (dimFilter) {
        const match = dimFilter.match(/(\d+)\s*x\s*(\d+)/i)
        if (match) {
          const r = parseInt(match[1], 10)
          const c = parseInt(match[2], 10)
          if (f.rows !== r || f.cols !== c) return false
        } else {
          // substring match
          if (!(`${f.rows}x${f.cols}`).includes(dimFilter.replace(/\s+/g, ''))) return false
        }
      }
      return true
    })
  }

  function FilterBar() {
    return (
      <div className="flex gap-3 flex-wrap">
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by name..." className="px-3 py-2 border rounded" />
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} className="px-3 py-2 border rounded">
          <option value="any">Any type</option>
          <option value="identity">Identity</option>
          <option value="symmetric">Symmetric</option>
          <option value="hessenberg">Hessenberg</option>
          <option value="general">General</option>
        </select>
        <select value={sparsityFilter} onChange={(e) => setSparsityFilter(e.target.value)} className="px-3 py-2 border rounded">
          <option value="any">Any density</option>
          <option value="sparse">Sparse (&lt;=20%)</option>
          <option value="medium">Medium (20-60%)</option>
          <option value="dense">Dense (&gt;=60%)</option>
        </select>
        <input value={dimFilter} onChange={(e) => setDimFilter(e.target.value)} placeholder="Dimension (e.g. 3x3)" className="px-3 py-2 border rounded" />
      </div>
    )
  }

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
                {/* search box kept small here -- main search in filter bar */}
              </div>
            </div>

            <div className="mb-6 flex flex-wrap gap-3 items-center">
              <FilterBar />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filtered().map((f) => {
                const origIdx = favorites.findIndex(fi => fi.matrixString === f.matrixString)
                return (
                  <div key={f.matrixString} onClick={() => { import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(f.matrixString)}/basic`)) }} className="bg-white border border-border-color rounded-xl p-5 shadow-sm hover:shadow-md hover:border-primary/30 transition-all group relative cursor-pointer">
                    <div className="absolute top-4 right-4 flex gap-2">
                      <button onClick={(e) => { e.stopPropagation(); removeFavorite(origIdx) }} className="text-slate-300 hover:text-red-500 transition-colors">
                        <span className="material-symbols-outlined text-[20px]">delete</span>
                      </button>
                      {editing.idx === origIdx ? (
                        <button onClick={(e) => { e.stopPropagation(); saveEdit() }} className="text-slate-400">Save</button>
                      ) : (
                        <button onClick={(e) => { e.stopPropagation(); startEdit(origIdx) }} className="text-slate-400">Edit</button>
                      )}
                    </div>

                    <div className="flex justify-center mb-6">
                      <div className="w-full flex justify-center">
                        <div className="w-32">
                          <MatrixPreview matrixString={f.matrixString} />
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5 mb-1">
                        <svg aria-hidden="true" className="w-[16px] h-[16px] text-yellow-400" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 .587l3.668 7.431L24 9.748l-6 5.847L19.335 24 12 20.201 4.665 24 6 15.595 0 9.748l8.332-1.73L12 .587z" />
                        </svg>
                        {editing.idx === origIdx ? (
                          <input className="font-bold text-slate-900 text-sm" value={editing.name} onChange={(e) => setEditing({ ...editing, name: e.target.value })} />
                        ) : (
                          <h3 className="font-bold text-slate-900 text-sm truncate">{f.name}</h3>
                        )}
                      </div>
                      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-50">
                        <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-tighter">{f.rows} x {f.cols}</span>
                        <span className="text-[10px] text-slate-400 italic">Added {new Date(f.ts).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
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

