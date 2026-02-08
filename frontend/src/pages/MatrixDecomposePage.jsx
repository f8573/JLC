import React, { useEffect, useMemo, useRef, useState } from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import MatrixLatex from '../components/matrix/MatrixLatex'
import MatrixActionBar, { MatrixFooterBar } from '../components/matrix/MatrixActionBar'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { usePrecisionUpdate } from '../hooks/usePrecisionUpdate'
import Latex from '../components/ui/Latex'
import AccuracyBadge from '../components/ui/AccuracyBadge'
import DecompositionDetailModal from '../components/ui/DecompositionDetailModal'
import { multiplyMatrices, transposeMatrix, createIdentityMatrix } from '../utils/matrixOperations'
import { formatNumber } from '../utils/format'

/**
 * Renders a basis set (collection of vectors) using set notation.
 *
 * @param {Object} props
 * @param {string} props.title
 * @param {number[][]} [props.vectors] - Matrix where each column is a basis vector
 */
function BasisSetSection({ title, vectors }) {
  const available = vectors && vectors.length > 0 && vectors[0]?.length > 0
  const numVectors = vectors?.[0]?.length || 0
  const dim = vectors?.length || 0

  // Format a single number for display
  const fmt = (val) => {
    if (val === undefined || val === null) return '0'
    if (typeof val === 'object' && ('real' in val || 'imag' in val)) {
      const r = val.real ?? 0
      const i = val.imag ?? 0
      if (Math.abs(i) < 1e-10) return formatNumber(r, 4)
      if (Math.abs(r) < 1e-10) return i === 1 ? 'i' : i === -1 ? '-i' : `${formatNumber(i, 4)}i`
      return `${formatNumber(r, 4)}${i >= 0 ? '+' : ''}${formatNumber(i, 4)}i`
    }
    return formatNumber(val, 4)
  }

  // Build LaTeX for basis set notation: \left\{ \begin{pmatrix} ... \end{pmatrix}, ... \right\}
  const buildBasisLatex = () => {
    if (!available) return ''
    const parts = []
    for (let col = 0; col < numVectors; col++) {
      const entries = []
      for (let row = 0; row < dim; row++) {
        entries.push(fmt(vectors[row][col]))
      }
      parts.push(`\\begin{pmatrix}${entries.join('\\\\\\\\')}\\end{pmatrix}`)
    }
    return `\\left\\{ ${parts.join(',\\, ')} \\right\\}`
  }

  return (
    <section className={`space-y-4 ${!available ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
        <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">{title}</h2>
        {available && (
          <span className="text-xs font-bold px-3 py-1 rounded-full text-primary bg-primary/5 border border-primary/10">
            <Latex tex={`\\text{dim}=${numVectors}`} />
          </span>
        )}
      </div>
      <div className="bg-slate-50/50 dark:bg-slate-800/60 p-6 rounded-xl border border-slate-100 dark:border-slate-700">
        {available ? (
          <div className="flex justify-center">
            <Latex tex={buildBasisLatex()} className="math-font text-base" />
          </div>
        ) : (
          <div className="text-center text-sm text-slate-400 dark:text-slate-500">Unavailable for this matrix</div>
        )}
      </div>
    </section>
  )
}

/**
 * Renders a decomposition section with optional factor matrices and accuracy badge.
 *
 * @param {Object} props
 * @param {string} props.title
 * @param {React.ReactNode} [props.formula]
 * @param {string} [props.formulaTex] - LaTeX formula for header and modal
 * @param {Array<{label: React.ReactNode, data?: number[][], w?: number, g?: number, s?: string, precision?: number}>} [props.data]
 * @param {string} [props.label]
 * @param {boolean} [props.available=true]
 * @param {Object} [props.validation] - Validation object from API
 * @param {number[][]} [props.originalMatrix] - Original matrix for detail modal
 * @param {number[][]} [props.reconstructedMatrix] - Reconstructed matrix for detail modal
 */
function DecompSection({ title, formula, formulaTex, data, label, available = true, validation, originalMatrix, reconstructedMatrix }) {
  const [modalOpen, setModalOpen] = useState(false)
  const layoutRef = useRef(null)
  const [columnCount, setColumnCount] = useState(1)
  const minCardWidth = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return 280
    const estimateForItem = (item) => {
      const matrix = item?.data
      const base = matrix?.data || matrix
      if (!Array.isArray(base) || base.length === 0 || !Array.isArray(base[0])) return 280
      const cols = base[0].length || 1
      // Keep matrices readable by scaling minimum card width with matrix column count.
      return Math.min(840, Math.max(240, 120 + cols * 56))
    }
    return data.reduce((max, item) => Math.max(max, estimateForItem(item)), 240)
  }, [data])
  useEffect(() => {
    if (!available || !Array.isArray(data) || data.length <= 1) {
      setColumnCount(1)
      return
    }
    const el = layoutRef.current
    if (!el) return
    const gapPx = 32
    const computeColumns = () => {
      const width = el.clientWidth || 0
      const fit = Math.floor((width + gapPx) / (minCardWidth + gapPx))
      const next = Math.max(1, Math.min(data.length, fit))
      setColumnCount(next)
    }
    computeColumns()
    const observer = new ResizeObserver(computeColumns)
    observer.observe(el)
    return () => observer.disconnect()
  }, [available, data, minCardWidth])

  return (
    <section className={`space-y-4 ${!available ? 'opacity-50' : ''}`}>
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">{title}</h2>
          {available && validation && (
            <AccuracyBadge 
              validation={validation} 
              compact={true}
              onInfoClick={() => setModalOpen(true)}
            />
          )}
        </div>
        {(formulaTex || formula) && (
          <span className={`text-xs font-bold px-3 py-1 rounded-full ${available ? 'text-primary bg-primary/5 dark:bg-primary/15 border border-primary/10 dark:border-primary/25' : 'text-slate-400 dark:text-slate-500 bg-slate-100 dark:bg-slate-700'}`}>
            {formulaTex ? <Latex tex={formulaTex} /> : formula}
          </span>
        )}
      </div>
      <div className="bg-slate-50/50 dark:bg-slate-800/60 p-6 rounded-xl border border-slate-100 dark:border-slate-700">
        {available && data && data.length > 0 ? (
          <div
            ref={layoutRef}
            className="grid gap-8"
            style={{ gridTemplateColumns: `repeat(${Math.max(1, columnCount)}, minmax(0, 1fr))` }}
          >
            {data.map((item, idx) => (
              <div key={idx} className="space-y-3">
                <p className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-widest text-center">{item.label}</p>
                <div className="flex justify-center">
                  {item.data ? (
                    <MatrixLatex data={item.data} className={`math-font ${item.s||'text-sm'} font-medium`} precision={item.precision || 2} />
                  ) : <div className="text-xs text-slate-400 dark:text-slate-500">—</div>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-sm text-slate-400 dark:text-slate-500">Unavailable for this matrix</div>
        )}
      </div>
      
      {/* Detail Modal */}
      <DecompositionDetailModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title={title}
        formula={formulaTex}
        validation={validation}
        originalMatrix={originalMatrix}
        reconstructedMatrix={reconstructedMatrix}
      />
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
  // Subscribe to precision changes
  usePrecisionUpdate()
  
  const { diagnostics } = useDiagnostics(matrixString)

  // Get original matrix data
  const originalMatrix = diagnostics?.matrixData?.data || diagnostics?.matrixData || null

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

  // Compute reconstructions for each decomposition
  const qrRecon = diagnostics?.qr?.q?.data && diagnostics?.qr?.r?.data
    ? multiplyMatrices(diagnostics.qr.q.data, diagnostics.qr.r.data)
    : null

  // For PA = LU, we validate by computing P^{-1}*L*U = P^T*L*U (since P is orthogonal)
  const luRecon = diagnostics?.lu?.l?.data && diagnostics?.lu?.u?.data
    ? (() => {
        const LU = multiplyMatrices(diagnostics.lu.l.data, diagnostics.lu.u.data)
        // P^{-1} = P^T for permutation matrices
        return diagnostics.lu.p?.data ? multiplyMatrices(transposeMatrix(diagnostics.lu.p.data), LU) : LU
      })()
    : null

  const cholRecon = diagnostics?.cholesky?.l?.data
    ? multiplyMatrices(diagnostics.cholesky.l.data, transposeMatrix(diagnostics.cholesky.l.data))
    : null

  const svdRecon = diagnostics?.svd?.u?.data && diagnostics?.svd?.sigma?.data && diagnostics?.svd?.v?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.svd.u.data, diagnostics.svd.sigma.data),
        transposeMatrix(diagnostics.svd.v.data)
      )
    : null

  const polarRecon = diagnostics?.polar?.u?.data && diagnostics?.polar?.p?.data
    ? multiplyMatrices(diagnostics.polar.u.data, diagnostics.polar.p.data)
    : null

  const hessRecon = diagnostics?.hessenbergDecomposition?.q?.data && diagnostics?.hessenbergDecomposition?.h?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.hessenbergDecomposition.q.data, diagnostics.hessenbergDecomposition.h.data),
        transposeMatrix(diagnostics.hessenbergDecomposition.q.data)
      )
    : null

  const schurRecon = diagnostics?.schurDecomposition?.u?.data && diagnostics?.schurDecomposition?.t?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.schurDecomposition.u.data, diagnostics.schurDecomposition.t.data),
        transposeMatrix(diagnostics.schurDecomposition.u.data)
      )
    : null

  const eigenRecon = diagnostics?.diagonalization?.p?.data && diagnostics?.diagonalization?.d?.data && diagnostics?.diagonalization?.pInverse?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.diagonalization.p.data, diagnostics.diagonalization.d.data),
        diagnostics.diagonalization.pInverse.data
      )
    : null

  const spectralRecon = diagnostics?.symmetricSpectral?.q?.data && diagnostics?.symmetricSpectral?.lambda?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.symmetricSpectral.q.data, diagnostics.symmetricSpectral.lambda.data),
        transposeMatrix(diagnostics.symmetricSpectral.q.data)
      )
    : null

  const bidiagRecon = diagnostics?.bidiagonalization?.u?.data && diagnostics?.bidiagonalization?.b?.data && diagnostics?.bidiagonalization?.v?.data
    ? multiplyMatrices(
        multiplyMatrices(diagnostics.bidiagonalization.u.data, diagnostics.bidiagonalization.b.data),
        transposeMatrix(diagnostics.bidiagonalization.v.data)
      )
    : null

  // For inverse validation: compute A * A^{-1} which should equal I
  const inverseMatrixObj = diagnostics?.inverseMatrix || null
  const inverseMatrix = inverseMatrixObj?.data || inverseMatrixObj || null
  const inverseRecon = originalMatrix && inverseMatrix
    ? multiplyMatrices(originalMatrix, inverseMatrix)
    : null
  const pseudoInverseMatrixObj = diagnostics?.pseudoInverseMatrix || diagnostics?.svd?.pseudoInverse || null
  const pseudoInverseMatrix = pseudoInverseMatrixObj?.data || pseudoInverseMatrixObj || null
  const pseudoInverseRecon = originalMatrix && pseudoInverseMatrix
    ? multiplyMatrices(multiplyMatrices(originalMatrix, pseudoInverseMatrix), originalMatrix)
    : null
  // The "expected" matrix for inverse validation is the identity
  const n = originalMatrix?.length || 0
  const identityMatrix = n > 0 ? createIdentityMatrix(n) : null

  const rowSpace = diagnostics?.rowSpaceBasis?.vectors ? transposeVectors(diagnostics.rowSpaceBasis.vectors) : null
  const colSpace = diagnostics?.columnSpaceBasis?.vectors ? transposeVectors(diagnostics.columnSpaceBasis.vectors) : null
  const nullSpace = diagnostics?.nullSpaceBasis?.vectors ? transposeVectors(diagnostics.nullSpaceBasis.vectors) : null

  return (
    <MatrixAnalysisLayout
      matrixString={matrixString}
      diagnostics={diagnostics}
      activeTab="decompose"
      title="Matrix Decompositions"
      subtitle="Factorizations and structural transformations"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Decompositions' }]} />}
      actions={<MatrixActionBar matrixString={matrixString} diagnostics={diagnostics} />}
    >
      <div className="p-8 space-y-8">
        <DecompSection 
          title="QR Decomposition" 
          formula="A = QR"
          formulaTex="A = QR"
          available={!!(diagnostics?.qr?.q?.data && diagnostics?.qr?.r?.data)} 
          validation={diagnostics?.qr?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={qrRecon}
          data={[
            { label: 'Q (Orthogonal)', data: diagnostics?.qr?.q },
            { label: 'R (Upper Triangular)', data: diagnostics?.qr?.r }
          ]} 
        />
        <DecompSection 
          title="LU Decomposition" 
          formula={diagnostics?.lu?.p?.data ? "PA = LU" : "A = LU"}
          formulaTex={diagnostics?.lu?.p?.data ? "P^{-1}LU = A" : "A = LU"}
          available={!!(diagnostics?.lu?.l?.data && diagnostics?.lu?.u?.data)} 
          validation={diagnostics?.lu?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={luRecon}
          data={[
            ...(diagnostics?.lu?.p?.data ? [{ label: 'P (Permutation)', data: diagnostics?.lu?.p }] : []),
            { label: 'L (Lower)', data: diagnostics?.lu?.l },
            { label: 'U (Upper)', data: diagnostics?.lu?.u }
          ]} 
        />
        <DecompSection 
          title="Cholesky Decomposition" 
          formula="A = LL^T"
          formulaTex="A = LL^T"
          available={!!(diagnostics?.cholesky?.l?.data)} 
          validation={diagnostics?.cholesky?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={cholRecon}
          data={[
            { label: 'L (Lower)', data: diagnostics?.cholesky?.l },
            { label: <Latex tex="L^T" />, data: diagnostics?.cholesky?.u }
          ]} 
        />
        <DecompSection 
          title="SVD" 
          formula="A = USV*"
          formulaTex="A = U\Sigma V^T"
          available={!!(diagnostics?.svd?.u?.data && diagnostics?.svd?.sigma?.data && diagnostics?.svd?.v?.data)} 
          validation={diagnostics?.svd?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={svdRecon}
          data={[
            { label: 'U', data: diagnostics?.svd?.u, w:24, g:6, s:'text-xs' },
            { label: 'S', data: diagnostics?.svd?.sigma, w:24, g:6, s:'text-xs' },
            { label: <Latex tex="V^*" />, data: diagnostics?.svd?.v, w:24, g:6, s:'text-xs' }
          ]} 
        />
        <DecompSection 
          title="Polar Decomposition" 
          formula="A = UH"
          formulaTex="A = UH"
          available={!!(diagnostics?.polar?.u?.data && diagnostics?.polar?.p?.data)} 
          validation={diagnostics?.polar?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={polarRecon}
          data={[
            { label: 'U (Unitary)', data: diagnostics?.polar?.u },
            { label: 'H (Hermitian)', data: diagnostics?.polar?.p }
          ]} 
        />
        <DecompSection 
          title="Hessenberg" 
          formula="A = QHQ^T"
          formulaTex="A = QHQ^T"
          available={!!(diagnostics?.hessenbergDecomposition?.q?.data && diagnostics?.hessenbergDecomposition?.h?.data)} 
          validation={diagnostics?.hessenbergDecomposition?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={hessRecon}
          data={[
            { label: 'Q', data: diagnostics?.hessenbergDecomposition?.q },
            { label: 'H', data: diagnostics?.hessenbergDecomposition?.h }
          ]} 
        />
        <DecompSection 
          title="Schur" 
          formula="A = UTU^T"
          formulaTex="A = UTU^T"
          available={!!(diagnostics?.schurDecomposition?.u?.data && diagnostics?.schurDecomposition?.t?.data)} 
          validation={diagnostics?.schurDecomposition?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={schurRecon}
          data={[
            { label: 'U', data: diagnostics?.schurDecomposition?.u },
            { label: 'T', data: diagnostics?.schurDecomposition?.t }
          ]} 
        />
        <DecompSection 
          title="Eigendecomposition" 
          formula="A = PDP^{-1}"
          formulaTex="A = PDP^{-1}"
          available={!!(diagnostics?.diagonalization?.p?.data && diagnostics?.diagonalization?.d?.data)} 
          validation={diagnostics?.diagonalization?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={eigenRecon}
          data={[
            { label: 'P', data: diagnostics?.diagonalization?.p },
            { label: 'D', data: diagnostics?.diagonalization?.d },
            { label: <Latex tex="P^{-1}" />, data: diagnostics?.diagonalization?.pInverse }
          ]} 
        />
        <DecompSection 
          title="Spectral (Symmetric)" 
          formula="A = Q?Q^T"
          formulaTex="A = Q\Lambda Q^T"
          available={!!(diagnostics?.symmetricSpectral?.q?.data && diagnostics?.symmetricSpectral?.lambda?.data)} 
          validation={diagnostics?.symmetricSpectral?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={spectralRecon}
          data={[
            { label: 'Q', data: diagnostics?.symmetricSpectral?.q },
            { label: '?', data: diagnostics?.symmetricSpectral?.lambda }
          ]} 
        />
        <DecompSection 
          title="Bidiagonalization" 
          formula="A = UBV^T"
          formulaTex="A = UBV^T"
          available={!!(diagnostics?.bidiagonalization?.u?.data && diagnostics?.bidiagonalization?.b?.data && diagnostics?.bidiagonalization?.v?.data)} 
          validation={diagnostics?.bidiagonalization?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={bidiagRecon}
          data={[
            { label: 'U', data: diagnostics?.bidiagonalization?.u, w:24, g:6, s:'text-xs' },
            { label: 'B', data: diagnostics?.bidiagonalization?.b, w:24, g:6, s:'text-xs' },
            { label: <Latex tex="V^T" />, data: diagnostics?.bidiagonalization?.v, w:24, g:6, s:'text-xs' }
          ]} 
        />
        <DecompSection 
          title="Inverse" 
          formula={<Latex tex="A \cdot A^{-1} = I" />}
          formulaTex="A \cdot A^{-1} = I"
          available={!!(diagnostics?.inverseMatrix?.data)} 
          validation={diagnostics?.inverse?.validation}
          originalMatrix={identityMatrix}
          reconstructedMatrix={inverseRecon}
          data={[
            { label: <Latex tex="A^{-1}" />, data: inverseMatrixObj }
          ]} 
        />
        <DecompSection
          title="Pseudo-inverse"
          formula={<Latex tex="A^+=U\Sigma^+ V^T" />}
          formulaTex="A^+=U\Sigma^+ V^T"
          available={!!pseudoInverseMatrix}
          validation={diagnostics?.pseudoInverse?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={pseudoInverseRecon}
          data={[
            { label: <Latex tex="A^{+}" />, data: pseudoInverseMatrixObj }
          ]}
        />
        <DecompSection 
          title="Reduced Row Echelon Form" 
          formula="RREF(A)"
          formulaTex="\text{RREF(A)}"
          available={!!(diagnostics?.rrefMatrix?.data)} 
          validation={diagnostics?.rref?.validation}
          originalMatrix={originalMatrix}
          reconstructedMatrix={null}
          data={[
            { label: 'RREF', data: diagnostics?.rrefMatrix }
          ]} 
        />
        <BasisSetSection title="Row Space Basis" vectors={rowSpace} />
        <BasisSetSection title="Column Space Basis" vectors={colSpace} />
        <BasisSetSection title="Null Space Basis" vectors={nullSpace} />
      </div>
      <MatrixFooterBar matrixString={matrixString} diagnostics={diagnostics} />
    </MatrixAnalysisLayout>
  )
}
