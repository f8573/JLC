import React from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { formatPercent } from '../utils/format'

/**
 * Status row describing a structural property of a matrix.
 *
 * @param {Object} props
 * @param {string} props.label
 * @param {boolean} props.active
 * @param {string} [props.detail]
 */
function StatusRow({ label, active, detail }) {
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
      active ? 'border-primary/20 bg-primary/5' : 'border-slate-100 hover:bg-slate-50'
    }`}>
      <div className="flex flex-col">
        <span className={`text-sm ${active ? 'font-bold text-primary' : 'font-medium'}`}>{label}</span>
        {detail && <span className="text-xs text-slate-400">{detail}</span>}
      </div>
      <span className={`material-symbols-outlined ${active ? 'text-primary text-[20px] font-bold' : 'text-slate-300'}`}>
        {active ? 'check_circle' : 'close'}
      </span>
    </div>
  )
}

/**
 * Structural analysis view that classifies matrices into named categories.
 *
 * @param {Object} props
 * @param {string} props.matrixString - Serialized matrix payload from the URL.
 */
export default function MatrixStructurePage({ matrixString }) {
  const { diagnostics } = useDiagnostics(matrixString)

  const basicStructure = [
    { label: 'Zero Matrix', active: diagnostics?.zero },
    { label: 'Identity Matrix', active: diagnostics?.identity },
    { label: 'Scalar Matrix', active: diagnostics?.scalar },
    { label: 'Diagonal', active: diagnostics?.diagonal },
    { label: 'Bidiagonal', active: diagnostics?.bidiagonal },
    { label: 'Tridiagonal', active: diagnostics?.tridiagonal }
  ]

  const triangularStructure = [
    { label: 'Upper Triangular', active: diagnostics?.upperTriangular },
    { label: 'Lower Triangular', active: diagnostics?.lowerTriangular },
    { label: 'Upper Hessenberg', active: diagnostics?.upperHessenberg },
    { label: 'Lower Hessenberg', active: diagnostics?.lowerHessenberg },
    { label: 'Hessenberg', active: diagnostics?.hessenberg },
    { label: 'Schur Form', active: diagnostics?.schur }
  ]

  const specialStructure = [
    { label: 'Block Diagonal', active: diagnostics?.block },
    { label: 'Companion Matrix', active: diagnostics?.companion }
  ]

  const symmetryStructure = [
    { label: 'Symmetric', active: diagnostics?.symmetric, detail: diagnostics?.symmetryError ? `error: ${diagnostics.symmetryError.toExponential(2)}` : null },
    { label: 'Skew-Symmetric', active: diagnostics?.skewSymmetric },
    { label: 'Hermitian', active: diagnostics?.hermitian },
    { label: 'Persymmetric', active: diagnostics?.persymmetric },
    { label: 'Antidiagonal', active: diagnostics?.antidiagonal }
  ]

  const algebraicStructure = [
    { label: 'Idempotent (A² = A)', active: diagnostics?.idempotent },
    { label: 'Involutory (A² = I)', active: diagnostics?.involutory },
    { label: 'Nilpotent (Aⁿ = 0)', active: diagnostics?.nilpotent }
  ]

  const geometricStructure = [
    { label: 'Rotation', active: diagnostics?.rotation },
    { label: 'Reflection', active: diagnostics?.reflection }
  ]

  const quickSummary = [
    { label: 'Rank', value: diagnostics?.rank ?? '—' },
    { label: 'Nullity', value: diagnostics?.nullity ?? '—' },
    {
      label: 'Singularity',
      value: diagnostics?.singular === null || diagnostics?.singular === undefined
        ? '—'
        : diagnostics.singular
          ? 'Singular'
          : 'Non-singular'
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
      activeTab="structure"
      title="Structural Properties"
      subtitle="Categorical classification and structural patterns"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Structure' }]} />}
      actions={
        <button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700">
          <span className="material-symbols-outlined text-[20px]">description</span>
          Export LaTeX
        </button>
      }
    >
      <div className="p-8 flex-1">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-8">
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Basic Structure</h4>
              <div className="flex flex-col gap-2">
                {basicStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Triangular & Hessenberg</h4>
              <div className="flex flex-col gap-2">
                {triangularStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Special Forms</h4>
              <div className="flex flex-col gap-2">
                {specialStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
          </div>
          <div className="space-y-8">
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Symmetry</h4>
              <div className="flex flex-col gap-2">
                {symmetryStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Algebraic Properties</h4>
              <div className="flex flex-col gap-2">
                {algebraicStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">Geometric Properties</h4>
              <div className="flex flex-col gap-2">
                {geometricStructure.map((row) => (
                  <StatusRow key={row.label} {...row} />
                ))}
              </div>
            </div>
            <div className="bg-slate-50 rounded-xl p-6 border border-slate-100 h-fit">
              <h4 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-tighter">Quick Summary</h4>
              <ul className="space-y-4">
                {quickSummary.map((item) => (
                  <li key={item.label} className="flex items-start gap-3">
                    <span className="material-symbols-outlined text-primary text-sm mt-1">check_circle</span>
                    <div>
                      <p className="text-sm font-bold">{item.label}</p>
                      <p className="text-xs text-slate-500">{item.value}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
      <footer className="flex items-center justify-between bg-slate-50 px-8 py-5 border-t border-slate-100">
        <div className="flex gap-3">
          <a href="#" className="text-xs font-bold hover:underline text-slate-400">↑ Back to Top</a>
          <a href="#" className="text-xs font-bold hover:underline text-slate-400">Matrix Dashboard</a>
        </div>
        <p className="text-xs text-slate-400">All property checks complete.</p>
      </footer>
    </MatrixAnalysisLayout>
  )
}

