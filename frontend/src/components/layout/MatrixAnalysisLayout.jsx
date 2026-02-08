import React, { useMemo } from 'react'
import Header from '../Header'
import Sidebar from '../Sidebar'
import MatrixTabs from './MatrixTabs'
import MatrixDisplay from '../matrix/MatrixDisplay'
import { parseMatrixString, analyzeAndCache, matrixToString } from '../../utils/diagnostics'
import { formatDimension, formatPercent } from '../../utils/format'

/**
 * Shared layout for all matrix analysis pages.
 *
 * @param {Object} props
 * @param {string} props.matrixString
 * @param {Object} [props.diagnostics]
 * @param {'basic'|'spectral'|'decompose'|'structure'} [props.activeTab='basic']
 * @param {string} props.title
 * @param {string} props.subtitle
 * @param {React.ReactNode} props.breadcrumbs
 * @param {React.ReactNode} props.actions
 * @param {React.ReactNode} props.children
 */
export default function MatrixAnalysisLayout({
  matrixString,
  diagnostics,
  activeTab = 'basic',
  title,
  subtitle,
  breadcrumbs,
  actions,
  children
}) {
  const matrixData = useMemo(() => {
    if (diagnostics?.matrixData) return diagnostics.matrixData
    return parseMatrixString(matrixString)
  }, [diagnostics, matrixString])
  const matrixRows = matrixData?.data || matrixData

  /**
   * Parse input, run diagnostics, and navigate to the basic results view.
   *
   * @param {string} inputValue
   * @returns {Promise<void>}
   */
  async function handleCompute(inputValue) {
    const parsed = parseMatrixString(inputValue)
    if (!parsed) {
      return
    }
    const normalizedString = matrixToString(parsed)
    try {
      await analyzeAndCache(parsed, normalizedString)
    } catch (err) {
      // allow navigation even if diagnostics fails
    }
    window.location.href = `/matrix=${encodeURIComponent(normalizedString)}/basic`
  }

  const rows = diagnostics?.rows ?? matrixRows?.length
  const cols = diagnostics?.columns ?? matrixRows?.[0]?.length
  const dimensionLabel = formatDimension(rows, cols)
  const domainLabel = diagnostics?.domain === 'C' ? 'C (Complex Numbers)' : 'R (Real Numbers)'
  const density = diagnostics?.density

  return (
    <div className="bg-background-light dark:bg-slate-900 font-display text-slate-900 dark:text-slate-100 h-screen overflow-hidden transition-colors duration-300">
      <Header inputValue={matrixString} onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="analysis" showCurrentAnalysis={true} />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light dark:bg-slate-900 transition-colors duration-300">
          <div className="max-w-[1800px] mx-auto p-8 space-y-6">
            <div className="flex flex-col gap-2">
              {breadcrumbs}
              <div className="flex flex-wrap justify-between items-end gap-4 mt-2">
                <div className="space-y-1">
                  <h1 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white uppercase">{title}</h1>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">{subtitle}</p>
                </div>
                <div className="flex gap-3">{actions}</div>
              </div>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
              <div className="xl:col-span-1">
                <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 shadow-sm sticky top-6 transition-colors duration-300">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest">Input Matrix A</h3>
                    <span className="text-[10px] font-bold text-primary bg-primary/10 px-2 py-0.5 rounded-full uppercase">
                      {dimensionLabel}
                    </span>
                  </div>
                  <div className="flex justify-center py-4">
                    <div className="relative p-6 border-l-2 border-r-2 border-slate-300 dark:border-slate-600">
                      <MatrixDisplay
                        data={matrixData}
                        minCellWidth={50}
                        gap={8}
                        className="matrix-grid math-font text-lg font-bold"
                        cellClassName="text-center p-2"
                        highlightDiagonal
                      />
                    </div>
                  </div>
                  <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-700 space-y-3">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-500 dark:text-slate-400">Domain</span>
                      <span className="text-slate-800 dark:text-slate-100">{domainLabel}</span>
                    </div>
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-slate-500 dark:text-slate-400">Density</span>
                      <span className="text-slate-800 dark:text-slate-100">
                        {density === null || density === undefined
                          ? '—'
                          : `${formatPercent(density, 1)} (${density >= 0.5 ? 'Dense' : 'Sparse'})`}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              <div className="xl:col-span-3">
                <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden flex flex-col min-h-[600px] transition-colors duration-300">
                  <MatrixTabs matrixString={matrixString} activeTab={activeTab} />
                  <div className="dark:[&_.bg-white]:bg-slate-800 dark:[&_.bg-slate-50]:bg-slate-800/60 dark:[&_.border-slate-100]:border-slate-700 dark:[&_.border-slate-200]:border-slate-700 dark:[&_.text-slate-900]:text-slate-100 dark:[&_.text-slate-800]:text-slate-100 dark:[&_.text-slate-700]:text-slate-200 dark:[&_.text-slate-600]:text-slate-300 dark:[&_.text-slate-500]:text-slate-400 dark:[&_.text-slate-400]:text-slate-500">
                    {children}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
