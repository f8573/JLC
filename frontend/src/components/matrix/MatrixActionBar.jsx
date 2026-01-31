import React, { useState } from 'react'
import FavoriteModal from '../ui/FavoriteModal'
import { useFavorites } from '../../hooks/useFavorites'
import { navigate } from '../../utils/navigation'

/**
 * Coming Soon modal for features not yet implemented.
 */
function ComingSoonModal({ open, onClose, title = 'Coming Soon' }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative bg-white rounded-xl shadow-lg p-8 max-w-md text-center">
        <div className="size-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="material-symbols-outlined text-primary text-[32px]">construction</span>
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">{title}</h3>
        <p className="text-slate-500 mb-6">This feature is currently under development and will be available soon.</p>
        <button
          onClick={onClose}
          className="px-6 py-2.5 bg-primary text-white rounded-xl text-sm font-bold hover:bg-primary/90 transition-all"
        >
          Got it
        </button>
      </div>
    </div>
  )
}

/**
 * JSON data modal for viewing and downloading diagnostics data.
 */
function JsonModal({ open, onClose, matrixString, diagnostics }) {
  if (!open) return null

  const handleDownload = (e) => {
    e.stopPropagation()
    const data = { matrixString, diagnostics }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `matrix-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative w-11/12 md:w-3/4 lg:w-2/3 max-h-[80vh] bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <h3 className="text-sm font-bold">JSON Data</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownload}
              className="text-xs font-semibold text-primary hover:text-primary/80 flex items-center gap-2"
            >
              <span className="material-symbols-outlined">download</span>
              Download
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); onClose() }}
              className="w-8 h-8 flex items-center justify-center text-white bg-red-500 rounded hover:bg-red-600"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>
        <div className="p-4 overflow-auto max-h-[72vh]">
          <pre className="whitespace-pre-wrap text-sm font-mono text-slate-800">
            {JSON.stringify({ matrixString, diagnostics }, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}

/**
 * Shared action bar component for all matrix analysis tabs.
 * Includes Export LaTeX, Full Report, JSON Data, and Favorite button.
 *
 * @param {Object} props
 * @param {string} props.matrixString - The matrix identifier
 * @param {Object} props.diagnostics - The diagnostics data for the matrix
 * @param {boolean} [props.showExportLatex=true] - Whether to show Export LaTeX button
 */
export default function MatrixActionBar({ matrixString, diagnostics, showExportLatex = true }) {
  const {
    isFavorited,
    favoriteModalOpen,
    favoriteDefaultName,
    savedMessage,
    openFavorite,
    saveFavorite,
    removeFavorite,
    cancelFavorite,
  } = useFavorites(matrixString, diagnostics)

  const [jsonModalOpen, setJsonModalOpen] = useState(false)
  const [latexModalOpen, setLatexModalOpen] = useState(false)

  const handleFullReport = () => {
    navigate(`/matrix=${encodeURIComponent(matrixString)}/report`)
  }

  return (
    <>
      <div className="flex items-center gap-3">
        {showExportLatex && (
          <button 
            onClick={() => setLatexModalOpen(true)}
            className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700"
          >
            <span className="material-symbols-outlined text-[20px]">description</span>
            Export LaTeX
          </button>
        )}
        <button
          onClick={handleFullReport}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700"
        >
          <span className="material-symbols-outlined text-[20px]">visibility</span>
          Full Report
        </button>
        <button
          onClick={() => setJsonModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700"
        >
          <span className="material-symbols-outlined text-[20px]">data_object</span>
          JSON Data
        </button>
        <button
          onClick={() => isFavorited ? removeFavorite() : openFavorite()}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700"
        >
          {isFavorited ? (
            <svg
              aria-hidden="true"
              className="w-[20px] h-[20px] text-yellow-400"
              viewBox="0 0 24 24"
              fill="currentColor"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M12 .587l3.668 7.431L24 9.748l-6 5.847L19.335 24 12 20.201 4.665 24 6 15.595 0 9.748l8.332-1.73L12 .587z" />
            </svg>
          ) : (
            <span className="material-symbols-outlined text-[20px]">star_border</span>
          )}
          {isFavorited ? 'Favorited' : 'Favorite'}
        </button>
      </div>

      <FavoriteModal
        open={favoriteModalOpen}
        defaultName={favoriteDefaultName}
        onCancel={cancelFavorite}
        onSave={saveFavorite}
      />

      <JsonModal
        open={jsonModalOpen}
        onClose={() => setJsonModalOpen(false)}
        matrixString={matrixString}
        diagnostics={diagnostics}
      />

      <ComingSoonModal
        open={latexModalOpen}
        onClose={() => setLatexModalOpen(false)}
        title="Export LaTeX"
      />

      {savedMessage && (
        <div className="fixed bottom-6 right-6 bg-black text-white px-4 py-2 rounded shadow z-50">
          {savedMessage}
        </div>
      )}
    </>
  )
}

/**
 * Footer bar with common actions (Full Report, JSON Data download).
 * Uses consistent routes across all tabs.
 *
 * @param {Object} props
 * @param {string} props.matrixString - The matrix identifier
 * @param {Object} props.diagnostics - The diagnostics data for the matrix
 */
export function MatrixFooterBar({ matrixString, diagnostics }) {
  const [jsonModalOpen, setJsonModalOpen] = useState(false)

  const handleViewReport = (e) => {
    e.preventDefault()
    navigate(`/matrix=${encodeURIComponent(matrixString)}/report`)
  }

  const handleDownloadJson = () => {
    const data = { matrixString, diagnostics }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `matrix-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <div className="bg-slate-50 border-t border-slate-200 px-8 py-4 flex items-center justify-between">
        <span className="text-xs text-slate-400 italic">Solved using high-precision JLA subroutines.</span>
        <div className="flex gap-6">
          <a
            className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5"
            href={`/matrix=${encodeURIComponent(matrixString)}/report`}
            onClick={handleViewReport}
          >
            <span className="material-symbols-outlined text-[18px]">visibility</span>
            Full Report
          </a>
          <button
            onClick={() => setJsonModalOpen(true)}
            className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5"
          >
            <span className="material-symbols-outlined text-[18px]">data_object</span>
            JSON Data
          </button>
        </div>
      </div>
      {jsonModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40" onClick={() => setJsonModalOpen(false)} />
          <div className="relative w-11/12 md:w-3/4 lg:w-2/3 max-h-[80vh] bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
              <h3 className="text-sm font-bold">JSON Data</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDownloadJson()
                  }}
                  className="text-xs font-semibold text-primary hover:text-primary/80 flex items-center gap-2"
                >
                  <span className="material-symbols-outlined">download</span>
                  Download
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); setJsonModalOpen(false) }}
                  className="w-8 h-8 flex items-center justify-center text-white bg-red-500 rounded hover:bg-red-600"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>
            <div className="p-4 overflow-auto max-h-[72vh]">
              <pre className="whitespace-pre-wrap text-sm font-mono text-slate-800">
                {JSON.stringify({ matrixString, diagnostics }, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
