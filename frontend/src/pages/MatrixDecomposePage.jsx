import React from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import MatrixDisplay from '../components/matrix/MatrixDisplay'
import { useDiagnostics } from '../hooks/useDiagnostics'
import Latex from '../components/ui/Latex'

/**
 * Renders a decomposition section with optional factor matrices.
 *
 * @param {Object} props
 * @param {string} props.title
 * @param {React.ReactNode} [props.formula]
 * @param {Array<{label: React.ReactNode, data?: number[][], w?: number, g?: number, s?: string}>} [props.data]
 * @param {string} [props.label]
 * @param {boolean} [props.available=true]
 */
function DecompSection({ title, formula, data, label, available = true }) {
  return (
    <section className={`space-y-4 ${!available ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-lg font-bold text-slate-800">{title}</h2>
        {formula && <span className={`text-xs font-bold px-3 py-1 rounded-full ${available ? 'text-primary bg-primary/5 border border-primary/10' : 'text-slate-400 bg-slate-100'}`}>{formula}</span>}
      </div>
      <div className="bg-slate-50/50 p-6 rounded-xl border border-slate-100">
        {available && data && data.length > 0 ? (
          <div className={`grid grid-cols-1 ${data.length > 2 ? 'md:grid-cols-3' : 'md:grid-cols-2'} gap-8`}>
            {data.map((item, idx) => (
              <div key={idx} className="space-y-3">
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest text-center">{item.label}</p>
                <div className="flex justify-center">
                  {item.data ? (
                    <div className="flex items-stretch gap-2">
                      <div className="matrix-bracket-left"></div>
                      <MatrixDisplay data={item.data} minCellWidth={item.w||30} gap={item.g||8} className={`math-font ${item.s||'text-sm'} font-medium`} cellClassName="text-center" />
                      <div className="matrix-bracket-right"></div>
                    </div>
                  ) : <div className="text-xs text-slate-400">—</div>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-sm text-slate-400">Unavailable for this matrix</div>
        )}
      </div>
    </section>
  )
}

/**
 * Decomposition overview page showing common factorization outputs.
 *
 * @param {Object} props
 * @param {string} props.matrixString - Serialized matrix payload from the URL.
 */
export default function MatrixDecomposePage({ matrixString }) {
  const { diagnostics } = useDiagnostics(matrixString)

  // Helper to transpose basis vectors for display
  /**
   * Transpose an array of basis vectors into column-major matrix layout.
   *
   * @param {number[][]} vecs
   * @returns {number[][] | null}
   */
  const transposeVectors = (vecs) => {
    if (!vecs || vecs.length === 0) return null
    const numRows = vecs[0].length
    const numCols = vecs.length
    const result = []
    for (let i = 0; i < numRows; i++) {
      const row = []
      for (let j = 0; j < numCols; j++) {
        row.push(vecs[j][i])
      }
      result.push(row)
    }
    return result
  }

  const rowSpace = diagnostics?.rowSpaceBasis?.value ? transposeVectors(Array.from(diagnostics.rowSpaceBasis.value).map(v => v.data)) : null
  const colSpace = diagnostics?.columnSpaceBasis?.value ? transposeVectors(Array.from(diagnostics.columnSpaceBasis.value).map(v => v.data)) : null
  const nullSpace = diagnostics?.nullSpaceBasis?.value ? transposeVectors(Array.from(diagnostics.nullSpaceBasis.value).map(v => v.data)) : null

  return (
    <MatrixAnalysisLayout
      matrixString={matrixString}
      diagnostics={diagnostics}
      activeTab="decompose"
      title="Matrix Decompositions"
      subtitle="Factorizations and structural transformations"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Decompositions' }]} />}
      actions={
        <button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700">
          <span className="material-symbols-outlined text-[20px]">description</span>
          Export LaTeX
        </button>
      }
    >
      <div className="p-8 space-y-8">
        <DecompSection title="QR Decomposition" formula="A = QR" available={diagnostics?.qr?.status === 'OK'} data={[
          { label: 'Q (Orthogonal)', data: diagnostics?.qr?.q?.data },
          { label: 'R (Upper Triangular)', data: diagnostics?.qr?.r?.data }
        ]} />
        <DecompSection title="LU Decomposition" formula="A = LU" available={diagnostics?.lu?.status === 'OK'} data={[
          { label: 'L (Lower)', data: diagnostics?.lu?.l?.data },
          { label: 'U (Upper)', data: diagnostics?.lu?.u?.data }
        ]} />
        <DecompSection title="Cholesky Decomposition" formula="A = LL^T" available={diagnostics?.cholesky?.status === 'OK'} data={[
          { label: 'L (Lower)', data: diagnostics?.cholesky?.l?.data },
          { label: <Latex tex="L^T" />, data: diagnostics?.cholesky?.u?.data }
        ]} />
        <DecompSection title="SVD" formula="A = UΣV*" available={diagnostics?.svd?.status === 'OK'} data={[
          { label: 'U', data: diagnostics?.svd?.u?.data, w:24, g:6, s:'text-xs' },
          { label: 'Σ', data: diagnostics?.svd?.sigma?.data, w:24, g:6, s:'text-xs' },
          { label: <Latex tex="V^*" />, data: diagnostics?.svd?.v?.data, w:24, g:6, s:'text-xs' }
        ]} />
        <DecompSection title="Polar Decomposition" formula="A = UH" available={diagnostics?.polar?.status === 'OK'} data={[
          { label: 'U (Unitary)', data: diagnostics?.polar?.u?.data },
          { label: 'H (Hermitian)', data: diagnostics?.polar?.p?.data }
        ]} />
        <DecompSection title="Hessenberg" formula="A = QHQ^T" available={diagnostics?.hessenbergDecomposition?.status === 'OK'} data={[
          { label: 'Q', data: diagnostics?.hessenbergDecomposition?.q?.data },
          { label: 'H', data: diagnostics?.hessenbergDecomposition?.h?.data }
        ]} />
        <DecompSection title="Schur" formula="A = UTU^T" available={diagnostics?.schurDecomposition?.status === 'OK'} data={[
          { label: 'U', data: diagnostics?.schurDecomposition?.u?.data },
          { label: 'T', data: diagnostics?.schurDecomposition?.t?.data }
        ]} />
        <DecompSection title="Eigendecomposition" formula="A = PDP^{-1}" available={diagnostics?.diagonalization?.status === 'OK'} data={[
          { label: 'P', data: diagnostics?.diagonalization?.p?.data },
          { label: 'D', data: diagnostics?.diagonalization?.d?.data },
          { label: <Latex tex="P^{-1}" />, data: diagnostics?.diagonalization?.pInverse?.data }
        ]} />
        <DecompSection title="Spectral (Symmetric)" formula="A = QΛQ^T" available={diagnostics?.symmetricSpectral?.status === 'OK'} data={[
          { label: 'Q', data: diagnostics?.symmetricSpectral?.q?.data },
          { label: 'Λ', data: diagnostics?.symmetricSpectral?.lambda?.data }
        ]} />
        <DecompSection title="Bidiagonalization" formula="A = UBV^T" available={diagnostics?.bidiagonalization?.status === 'OK'} data={[
          { label: 'U', data: diagnostics?.bidiagonalization?.u?.data, w:24, g:6, s:'text-xs' },
          { label: 'B', data: diagnostics?.bidiagonalization?.b?.data, w:24, g:6, s:'text-xs' },
          { label: <Latex tex="V^T" />, data: diagnostics?.bidiagonalization?.v?.data, w:24, g:6, s:'text-xs' }
        ]} />
        <DecompSection title="Inverse" formula={<Latex tex="A^{-1}" />} available={diagnostics?.inverse?.status === 'OK'} data={[
          { label: <Latex tex="A^{-1}" />, data: diagnostics?.inverseMatrix?.data }
        ]} />
        <DecompSection title="Reduced Row Echelon Form" formula="RREF(A)" available={diagnostics?.rref?.status === 'OK'} data={[
          { label: 'RREF', data: diagnostics?.rrefMatrix?.data }
        ]} />
        <DecompSection title="Row Space Basis" available={!!rowSpace} data={[
          { label: `Basis (dim: ${rowSpace?.[0]?.length || 0})`, data: rowSpace }
        ]} />
        <DecompSection title="Column Space Basis" available={!!colSpace} data={[
          { label: `Basis (dim: ${colSpace?.[0]?.length || 0})`, data: colSpace }
        ]} />
        <DecompSection title="Null Space Basis" available={!!nullSpace} data={[
          { label: `Basis (dim: ${nullSpace?.[0]?.length || 0})`, data: nullSpace }
        ]} />
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

