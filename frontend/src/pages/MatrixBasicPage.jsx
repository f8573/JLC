import React from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import PropertyCard from '../components/results/PropertyCard'
import SummaryItem from '../components/results/SummaryItem'
import MatrixDisplay from '../components/matrix/MatrixDisplay'
import Latex from '../components/ui/Latex'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { formatNumber, formatDimension, formatPercent } from '../utils/format'

export default function MatrixBasicPage({ matrixString }) {
  const { diagnostics } = useDiagnostics(matrixString)

  const rows = diagnostics?.rows
  const cols = diagnostics?.columns
  const dimensionLabel = formatDimension(rows, cols)

  const properties = [
    {
      icon: 'straighten',
      label: 'Dimensions & Domain',
      value: (
        <div>
          <div>{dimensionLabel}</div>
          <div className="text-xs">
            <div>
              <Latex tex={diagnostics?.domain === 'C' ? '\\mathbb{C}\\text{ (Complex)}' : '\\mathbb{R}\\text{ (Real)}'} />
            </div>
          </div>
        </div>
      ),
      iconBg: 'bg-primary/10',
      iconColor: 'text-primary'
    },
    {
      icon: 'square_foot',
      label: 'Norms',
      value: (
        <div>
          <div><Latex tex={`\\|A\\|_1 = ${formatNumber(diagnostics?.norm1, 4)}`} /></div>
          <div><Latex tex={`\\|A\\|_\\infty = ${formatNumber(diagnostics?.normInf, 4)}`} /></div>
          <div><Latex tex={`\\|A\\|_F = ${formatNumber(diagnostics?.frobeniusNorm, 4)}`} /> </div>
        </div>
      ),
      iconBg: 'bg-accent-purple/10',
      iconColor: 'text-accent-purple'
    },
    {
      icon: 'grid_on',
      label: 'Rank & Nullity',
      value: (
        <div>
          <div><Latex tex={`\\text{rank} = ${diagnostics?.rank ?? '—'}`} /></div>
          <div><Latex tex={`\\text{nullity} = ${diagnostics?.nullity ?? '—'}`} /></div>
        </div>
      ),
      iconBg: 'bg-purple-500/10',
      iconColor: 'text-purple-500'
    },
    {
      icon: 'shield',
      label: 'Invertibility & Conditioning',
      value: (
        <div>
          <div>
            {diagnostics?.invertible
              ? <Latex tex={'\\textbf{Invertible}'} className="inline-block text-emerald-600" />
              : diagnostics?.singular
                ? <Latex tex={'\\textbf{Singular}'} className="inline-block text-rose-600" />
                : '—'
            }
          </div>
          <div className="text-xs">
            {diagnostics?.conditionNumber
              ? <Latex tex={`\\kappa\\approx ${formatNumber(diagnostics.conditionNumber, 3)}`.replace(/\\\\/g,'\\')} />
              : '—'}
          </div>
        </div>
      ),
      iconBg: 'bg-emerald-100',
      iconColor: 'text-emerald-600'
    },
    {
      icon: 'calculate',
      label: 'Scalar Invariants',
      value: (
        <div>
          <div><Latex tex={`\\text{trace} = ${formatNumber(diagnostics?.trace, 4)}`} /></div>
          <div><Latex tex={`\\text{det}(A) = ${formatNumber(diagnostics?.determinant, 4)}`} /></div>
        </div>
      ),
      iconBg: 'bg-sky-100',
      iconColor: 'text-sky-600'
    },
    {
      icon: 'format_align_left',
      label: 'Row Reduction',
      value: (
        <div>
          <div>
            <span className="mr-2"><Latex tex={'\\text{RREF} '} /></span>
            {(() => {
              const item = diagnostics?.rref
              const ok = item && item.status === 'OK'
              return (
                <span className={ok ? 'text-emerald-600 font-semibold' : 'text-rose-600'}>
                  <Latex tex={ok ? '\\text{OK}' : '\\text{—}'} />
                </span>
              )
            })()}
          </div>
          <div className="text-xs">Row echelon: {diagnostics?.rowEchelon ? 'Yes' : 'No'}</div>
        </div>
      ),
      iconBg: 'bg-amber-100',
      iconColor: 'text-amber-600'
    }
  ]

  const summaryItems = [
    {
      title: diagnostics?.fullRank ? 'Full Rank' : 'Rank Deficient',
      description: diagnostics?.fullRank
        ? 'The matrix spans the full dimension of its column space.'
        : 'The matrix does not span the full dimension of its column space.'
    },
    {
      title: diagnostics?.invertible
        ? <Latex tex={'\\textbf{Invertible}'} className="inline-block text-emerald-600" />
        : diagnostics?.singular
          ? <Latex tex={'\\textbf{Singular}'} className="inline-block text-rose-600" />
          : 'Not Invertible',
      description: diagnostics?.invertible
        ? 'Determinant is non-zero, so an inverse exists.'
        : 'Determinant is zero or near-zero, inverse is not available.'
    },
    {
      title: 'Condition Number',
      description: diagnostics?.conditionNumber
        ? <Latex tex={`\\kappa\\approx ${formatNumber(diagnostics?.conditionNumber, 3)}`} />
        : 'Condition number unavailable.'
    }
  ]

  const spectralPreview = [
    { label: 'Spectral Radius', value: formatNumber(diagnostics?.spectralRadius, 4) },
    { label: 'Operator Norm', value: formatNumber(diagnostics?.operatorNorm, 4) },
    { label: 'Condition #', value: formatNumber(diagnostics?.conditionNumber, 3) },
    {
      label: 'Defectivity',
      value: diagnostics?.defectivity === null || diagnostics?.defectivity === undefined
        ? '—'
        : diagnostics.defectivity
          ? 'Defective'
          : 'Non-defective',
      color: diagnostics?.defectivity ? 'text-rose-500' : 'text-emerald-600'
    }
  ]

  const qrQ = diagnostics?.qr?.q?.data
  const qrR = diagnostics?.qr?.r?.data

  const structurePreview = [
    { label: 'Rank', value: diagnostics?.rank ?? '—' },
    { label: 'Nullity', value: diagnostics?.nullity ?? '—' },
    {
      label: 'Singularity',
      value: diagnostics?.singular === null || diagnostics?.singular === undefined
        ? '—'
        : diagnostics.singular
          ? <Latex tex={'\\textbf{singular}'} className="inline-block text-rose-600" />
          : <Latex tex={'\\text{Non-singular}'} className="inline-block text-emerald-600" />
    },
    {
      label: 'Density',
      value: diagnostics?.density === null || diagnostics?.density === undefined
        ? '—'
        : formatPercent(diagnostics.density, 1)
    }
  ]

  return (
    <MatrixAnalysisLayout
      matrixString={matrixString}
      diagnostics={diagnostics}
      activeTab="basic"
      title="Analysis Results"
      subtitle="Real-time property inspection and categorical decomposition"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Analysis Results' }]} />}
      actions={
        <button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700">
          <span className="material-symbols-outlined text-[20px]">description</span>
          Export LaTeX
        </button>
      }
    >
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

        <div className="mt-12">
          <div className="flex items-center gap-2 mb-4">
            <div className="h-[1px] flex-1 bg-slate-200"></div>
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Navigation Preview</span>
            <div className="h-[1px] flex-1 bg-slate-200"></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 border border-dashed border-slate-200 rounded-lg">
              <h5 className="text-[10px] font-bold mb-3 uppercase text-slate-400">Spectral Analysis</h5>
              <div className="space-y-2">
                {spectralPreview.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-slate-500">{item.label}</span>
                    <span className={`font-semibold ${item.color || 'text-slate-700'}`}>{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-4 border border-dashed border-slate-200 rounded-lg">
              <h5 className="text-[10px] font-bold mb-3 uppercase text-slate-400">Decompositions</h5>
              <div className="grid grid-cols-2 gap-3 text-[10px] text-slate-500">
                <div>
                  <p className="mb-1 text-[10px] font-bold uppercase text-slate-400">Q</p>
                  <div className="border border-slate-100 rounded p-2 bg-slate-50">
                    <MatrixDisplay
                      data={qrQ}
                      minCellWidth={18}
                      gap={4}
                      className="text-[9px]"
                      cellClassName="text-center"
                    />
                  </div>
                </div>
                <div>
                  <p className="mb-1 text-[10px] font-bold uppercase text-slate-400">R</p>
                  <div className="border border-slate-100 rounded p-2 bg-slate-50">
                    <MatrixDisplay
                      data={qrR}
                      minCellWidth={18}
                      gap={4}
                      className="text-[9px]"
                      cellClassName="text-center"
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="p-4 border border-dashed border-slate-200 rounded-lg">
              <h5 className="text-[10px] font-bold mb-3 uppercase text-slate-400">Structure</h5>
              <div className="space-y-2">
                {structurePreview.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-slate-500">{item.label}</span>
                    <span className="font-semibold text-slate-700">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="bg-slate-50 border-t border-slate-200 px-8 py-4 flex items-center justify-between">
        <span className="text-xs text-slate-400 italic">Solved using high-precision JLA subroutines.</span>
        <div className="flex gap-6">
          <a
            className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5"
            href={`/matrix=${encodeURIComponent(matrixString)}/report`}
            onClick={(e) => { e.preventDefault(); import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(matrixString)}/report`)) }}
          >
            <span className="material-symbols-outlined text-[18px]">visibility</span>
            Full Report
          </a>
          <button className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[18px]">download</span>
            JSON Data
          </button>
        </div>
      </div>
    </MatrixAnalysisLayout>
  )
}

