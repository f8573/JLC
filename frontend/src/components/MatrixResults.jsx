import React from 'react'
import Breadcrumb from './results/Breadcrumb'
import PropertyCard from './results/PropertyCard'
import SummaryItem from './results/SummaryItem'
import Button from './ui/Button'
import Badge from './ui/Badge'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { formatNumber, formatDimension, formatPercent } from '../utils/format'

const breadcrumbItems = [
  { label: 'Dashboard', href: '#' },
  { label: 'Analysis Results' }
]

const properties = [
  { icon: 'straighten', label: 'Dimensions', value: '3 × 3', iconBg: 'bg-primary/10', iconColor: 'text-primary' },
  { icon: 'change_history', label: 'Determinant', value: '-217', iconBg: 'bg-accent-purple/10', iconColor: 'text-accent-purple' },
  { icon: 'line_weight', label: 'Trace', value: '9', iconBg: 'bg-purple-500/10', iconColor: 'text-purple-500' }
]

const summaryItems = [
  { title: 'Full Rank', description: 'The matrix is non-singular and spans R³.' },
  { title: 'Invertible', description: 'Determinant is non-zero (∆ ≠ 0).' },
  { title: 'Condition Number', description: 'κ ≈ 18.42 (Well-conditioned).' }
]

/**
 * Analysis summary view for the matrix dashboard.
 */
export default function MatrixResults() {
  const { diagnostics } = useDiagnostics(window.location.pathname.startsWith('/matrix=') ? decodeURIComponent((window.location.pathname.match(/^\/matrix=([^/]+)/) || [])[1] || '') : '')

  const rows = diagnostics?.rows
  const cols = diagnostics?.columns
  const dimensionLabel = formatDimension(rows, cols)

  const properties = [
    { icon: 'straighten', label: 'Dimensions', value: dimensionLabel, iconBg: 'bg-primary/10', iconColor: 'text-primary' },
    { icon: 'change_history', label: 'Determinant', value: formatNumber(diagnostics?.determinant, 4), iconBg: 'bg-accent-purple/10', iconColor: 'text-accent-purple' },
    { icon: 'line_weight', label: 'Trace', value: formatNumber(diagnostics?.trace, 4), iconBg: 'bg-purple-500/10', iconColor: 'text-purple-500' }
  ]

  const summaryItems = [
    { title: diagnostics?.fullRank ? 'Full Rank' : 'Rank Deficient', description: diagnostics?.fullRank ? 'The matrix is non-singular and spans its column space.' : 'The matrix does not span the full column space.' },
    { title: diagnostics?.invertible ? 'Invertible' : 'Not Invertible', description: diagnostics?.invertible ? 'Determinant is non-zero (inverse exists).' : 'Determinant is zero or near-zero.' },
    { title: 'Condition Number', description: diagnostics?.conditionNumber ? `κ ≈ ${formatNumber(diagnostics.conditionNumber, 3)}` : 'Condition number unavailable.' }
  ]

  return (
    <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light">
      <div className="max-w-[1200px] mx-auto p-8 space-y-6">
        <div className="flex flex-col gap-2">
          <Breadcrumb items={breadcrumbItems} />
          <div className="flex flex-wrap justify-between items-end gap-4 mt-2">
            <div className="space-y-1">
              <h1 className="text-3xl font-black tracking-tight text-slate-900 uppercase">Analysis Results</h1>
              <p className="text-slate-500 text-sm">Real-time property inspection and categorical decomposition</p>
            </div>
            <div className="flex gap-3">
              <Button variant="secondary" className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm">
                <span className="material-symbols-outlined text-[20px]">description</span>
                Export LaTeX
              </Button>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <div className="xl:col-span-1">
            <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm sticky top-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Input Matrix A</h3>
                <Badge variant="status">3 × 3</Badge>
              </div>
              <div className="flex justify-center py-4">
                <div className="relative p-6 border-l-2 border-r-2 border-slate-300">
                  <div className="matrix-grid math-font text-lg font-bold">
                    <div className="text-center p-2 purple-glow">4</div>
                    <div className="text-center p-2">7</div>
                    <div className="text-center p-2">2</div>
                    <div className="text-center p-2">1</div>
                    <div className="text-center p-2 purple-glow">0</div>
                    <div className="text-center p-2">8</div>
                    <div className="text-center p-2">3</div>
                    <div className="text-center p-2">9</div>
                    <div className="text-center p-2 purple-glow">5</div>
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-slate-100 space-y-3">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-500">Domain</span>
                  <span className="text-slate-800">R (Real Numbers)</span>
                </div>
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-slate-500">Density</span>
                  <span className="text-slate-800">100% (Dense)</span>
                </div>
              </div>
            </div>
          </div>
          <div className="xl:col-span-3">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col min-h-[600px]">
              <div className="flex border-b border-slate-200 bg-slate-50/50">
                <button className="flex-1 px-4 py-4 text-xs font-bold uppercase tracking-widest border-b-2 transition-all tab-active">
                  Basic Properties
                </button>
                <button className="flex-1 px-4 py-4 text-xs font-bold uppercase tracking-widest border-b-2 transition-all tab-inactive">
                  Spectral Analysis
                </button>
                <button className="flex-1 px-4 py-4 text-xs font-bold uppercase tracking-widest border-b-2 transition-all tab-inactive">
                  Matrix Decompositions
                </button>
                <button className="flex-1 px-4 py-4 text-xs font-bold uppercase tracking-widest border-b-2 transition-all tab-inactive">
                  Structure
                </button>
              </div>
              <div className="p-8 flex-1">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="space-y-6">
                    {properties.map((prop, idx) => (
                      <PropertyCard key={idx} {...prop} />
                    ))}
                  </div>
                  <div className="bg-slate-50 rounded-xl p-6 border border-slate-100">
                    <h4 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-tighter">Quick Summary</h4>
                    <ul className="space-y-4">
                      {summaryItems.map((item, idx) => (
                        <SummaryItem key={idx} {...item} />
                      ))}
                    </ul>
                  </div>
                </div>
                <div className="mt-12 opacity-60 select-none">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="h-[1px] flex-1 bg-slate-200"></div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Navigation Preview</span>
                    <div className="h-[1px] flex-1 bg-slate-200"></div>
                  </div>
                  <div className="grid grid-cols-3 gap-6">
                    <div className="p-4 border border-dashed border-slate-200 rounded-lg">
                      <h5 className="text-[10px] font-bold mb-2 uppercase text-slate-400">Spectral Analysis</h5>
                      <div className="h-1.5 w-3/4 bg-slate-100 rounded mb-2"></div>
                      <div className="h-1.5 w-1/2 bg-slate-100 rounded"></div>
                    </div>
                    <div className="p-4 border border-dashed border-slate-200 rounded-lg">
                      <h5 className="text-[10px] font-bold mb-2 uppercase text-slate-400">Decompositions</h5>
                      <div className="h-1.5 w-2/3 bg-slate-100 rounded mb-2"></div>
                      <div className="h-1.5 w-full bg-slate-100 rounded"></div>
                    </div>
                    <div className="p-4 border border-dashed border-slate-200 rounded-lg">
                      <h5 className="text-[10px] font-bold mb-2 uppercase text-slate-400">Structure</h5>
                      <div className="h-1.5 w-1/2 bg-slate-100 rounded mb-2"></div>
                      <div className="h-1.5 w-3/4 bg-slate-100 rounded"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-slate-50 border-t border-slate-200 px-8 py-4 flex items-center justify-between">
                <span className="text-xs text-slate-400 italic">Solved using high-precision JLA subroutines.</span>
                <div className="flex gap-6">
                  <button className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-[18px]">visibility</span>
                    Full Report
                  </button>
                  <button className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-[18px]">download</span>
                    JSON Data
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
