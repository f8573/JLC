import React, { useEffect, useState, useMemo } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { parseMatrixString } from '../utils/diagnostics'
import MatrixLatex from '../components/matrix/MatrixLatex'

function MatrixPreviewSmall({ matrixString }) {
  const matrix = parseMatrixString(matrixString) || []
  const rows = Math.min(4, matrix.length)
  const cols = Math.min(4, (matrix[0] || []).length)
  if (rows === 0) return <div className="px-3 py-2 bg-slate-50 dark:bg-slate-700 rounded-sm">Empty</div>
  const previewMatrix = matrix.slice(0, rows).map(row => row.slice(0, cols))
  return (
    <div className="px-3 py-2 bg-slate-50 dark:bg-slate-700 rounded-sm">
      <MatrixLatex data={previewMatrix} className="text-[10px] math-font text-slate-600 dark:text-slate-300" precision={2} />
    </div>
  )
}

const ITEMS_PER_PAGE = 10

/**
 * History view listing completed matrix sessions and actions.
 */
export default function HistoryPage() {
  const [history, setHistory] = useState([])
  const [search, setSearch] = useState('')
  const [visibleCount, setVisibleCount] = useState(ITEMS_PER_PAGE)

  const loadHistory = () => {
    try {
      const raw = localStorage.getItem('recentSessions')
      const arr = raw ? JSON.parse(raw) : []
      // Sort by timestamp descending (most recent first)
      arr.sort((a, b) => (b.ts || 0) - (a.ts || 0))
      setHistory(arr)
    } catch (e) {
      setHistory([])
    }
  }

  useEffect(() => {
    loadHistory()
    
    // Listen for history-cleared events from settings
    const handleHistoryCleared = () => {
      setHistory([])
    }
    window.addEventListener('history-cleared', handleHistoryCleared)
    return () => window.removeEventListener('history-cleared', handleHistoryCleared)
  }, [])

  // Filter history based on search term
  const filteredHistory = useMemo(() => {
    if (!search.trim()) return history
    const searchLower = search.toLowerCase()
    return history.filter(h => {
      const title = (h.title || '').toLowerCase()
      const dimensions = `${h.rows || ''}x${h.cols || ''}`.toLowerCase()
      const date = h.ts ? new Date(h.ts).toLocaleString().toLowerCase() : ''
      return title.includes(searchLower) || dimensions.includes(searchLower) || date.includes(searchLower)
    })
  }, [history, search])

  // Get visible items based on pagination
  const visibleHistory = useMemo(() => {
    return filteredHistory.slice(0, visibleCount)
  }, [filteredHistory, visibleCount])

  const hasMore = visibleCount < filteredHistory.length

  function handleLoadMore() {
    setVisibleCount(prev => prev + ITEMS_PER_PAGE)
  }

  // Reset visible count when search changes
  useEffect(() => {
    setVisibleCount(ITEMS_PER_PAGE)
  }, [search])

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#ffffff] dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans selection:bg-primary/20 transition-colors duration-300">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active="history" />
        <main className="flex-1 overflow-y-auto bg-white dark:bg-slate-900 math-grid relative flex flex-col transition-colors duration-300">
          <div className="p-8 max-w-7xl mx-auto w-full">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
              <div>
                <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight">Computation History</h1>
                <p className="text-sm text-slate-500 dark:text-slate-400">Review and restore your past matrix operations.</p>
              </div>
              <div className="w-full md:w-80">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400 dark:text-slate-500">
                    <span className="material-symbols-outlined text-[20px]">search</span>
                  </div>
                  <input
                    className="block w-full bg-white dark:bg-slate-700 border border-border-color dark:border-slate-600 rounded-lg py-2 pl-10 pr-3 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all shadow-sm"
                    placeholder="Search history..."
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {/* Results count */}
            {search && (
              <div className="mb-4 text-sm text-slate-500 dark:text-slate-400">
                Found {filteredHistory.length} result{filteredHistory.length !== 1 ? 's' : ''} for "{search}"
              </div>
            )}

            <div className="space-y-3">
              {visibleHistory.length === 0 && (
                <div className="text-slate-400 dark:text-slate-500 text-center py-8">
                  {search ? 'No matching sessions found' : 'No history available'}
                </div>
              )}
              {visibleHistory.map((h, idx) => (
                <div key={h.ts || idx} onClick={() => { import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(h.title)}/basic`)) }} className="bg-white dark:bg-slate-800 border border-border-color dark:border-slate-700 rounded-xl p-4 flex items-center gap-6 hover:border-primary/30 transition-all group shadow-sm cursor-pointer">
                  <div className="w-32 flex justify-center shrink-0">
                    <MatrixPreviewSmall matrixString={h.title} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-bold text-slate-900 dark:text-white truncate">{(h.title || '').slice(0, 60)}</h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Operation: <span className="text-primary font-medium">Viewed</span></p>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="text-[11px] font-medium text-slate-400 dark:text-slate-500 block mb-1">{h.ts ? new Date(h.ts).toLocaleString() : ''}</span>
                    <div className="flex items-center gap-2 justify-end">
                      <span className="text-[10px] font-mono font-bold text-slate-400 dark:text-slate-500 bg-slate-100 dark:bg-slate-700 px-1.5 py-0.5 rounded">{h.rows || ''}x{h.cols || ''}</span>
                    </div>
                  </div>
                  <div className="pl-4 border-l border-slate-100 dark:border-slate-700 shrink-0">
                    <button onClick={(e) => { e.stopPropagation(); import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(h.title)}/basic`)) }} className="flex items-center gap-1.5 px-4 py-1.5 text-primary bg-purple-light dark:bg-primary/20 hover:bg-primary hover:text-white text-xs font-bold rounded-lg transition-all">
                      <span className="material-symbols-outlined text-[18px]">restore</span>
                      Restore
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {hasMore && (
              <div className="mt-8 flex justify-center">
                <button 
                  onClick={handleLoadMore}
                  className="text-sm font-semibold text-slate-500 dark:text-slate-400 hover:text-primary transition-colors flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-primary/5 dark:hover:bg-primary/10"
                >
                  <span>Load more sessions ({filteredHistory.length - visibleCount} remaining)</span>
                  <span className="material-symbols-outlined text-[20px]">expand_more</span>
                </button>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

