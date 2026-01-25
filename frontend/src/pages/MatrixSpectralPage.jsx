import React from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import MatrixDisplay from '../components/matrix/MatrixDisplay'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { formatComplex, formatNumber } from '../utils/format'
import Latex from '../components/ui/Latex'
import { computeSpectralSeverity, computePerEigenvalueSeverity, SEVERITY_COLORS } from '../utils/spectralSeverity'

function formatPolynomial(coeffs) {
  if (!Array.isArray(coeffs) || coeffs.length === 0) return '—'
  const degree = coeffs.length - 1
  const terms = []
  coeffs.forEach((coeff, idx) => {
    const power = degree - idx
    const real = coeff?.real ?? 0
    const imag = coeff?.imag ?? 0
    const magnitude = Math.hypot(real, imag)
    if (magnitude < 1e-10) return
    let coeffStr = formatComplex(coeff, 3)
    if (power > 0 && (coeffStr === '1' || coeffStr === '1.0')) coeffStr = ''
    if (power > 0 && (coeffStr === '-1' || coeffStr === '-1.0')) coeffStr = '-'
    const lambdaPart = power === 0 ? '' : power === 1 ? '\\lambda' : `\\lambda^${power}`
    terms.push(`${coeffStr}${lambdaPart}`.trim())
  })
  return terms.length ? `p(\\lambda) = ${terms.join(' + ').replace(/\+ -/g, '- ')}` : '—'
}

export default function MatrixSpectralPage({ matrixString }) {
  const { diagnostics } = useDiagnostics(matrixString)
  const eigenvalues = diagnostics?.eigenvalues || []
  const alg = diagnostics?.algebraicMultiplicity || []
  const geo = diagnostics?.geometricMultiplicity || []

  const findOriginalIndex = (lambda, tol = 1e-8) => {
    if (!lambda || !eigenvalues || eigenvalues.length === 0) return -1
    for (let i = 0; i < eigenvalues.length; i++) {
      const ev = eigenvalues[i]
      if (Math.hypot((ev?.real ?? 0) - (lambda?.real ?? 0), (ev?.imag ?? 0) - (lambda?.imag ?? 0)) <= tol) return i
    }
    return -1
  }

  const overallSeverity = computeSpectralSeverity(diagnostics)
  const criticalArtifacts = overallSeverity?.criticalArtifacts || {}
  const eigenvectors = diagnostics?.eigenvectors?.data
  // Eigenspace is meaningful only when the matrix is diagonalizable (non-defective).
  // If backend did not provide `eigenspace` for defective matrices this will be null.
  const eigenspace = diagnostics?.diagonalizable ? diagnostics?.eigenspace?.data : null
  const perEigenInfo = diagnostics?.eigenInformationPerValue || []
  const tol = 1e-8
  const uniqueEigenvalues = (() => {
    const u = []
    for (let k = 0; k < eigenvalues.length; k++) {
      const ev = eigenvalues[k]
      let found = false
      for (const x of u) {
        if (Math.hypot((x?.real ?? 0) - (ev?.real ?? 0), (x?.imag ?? 0) - (ev?.imag ?? 0)) <= tol) { found = true; break }
      }
      if (!found) u.push(ev)
    }
    return u
  })()
  const diagD = diagnostics?.diagonalization?.d || diagnostics?.d
  const diagP = diagnostics?.diagonalization?.p || diagnostics?.p
  const diagPInverse = diagnostics?.diagonalization?.pInverse
  const diagStatus = diagnostics?.diagonalization?.status || diagnostics?.diagonalization?.message || null
  const singularValues = diagnostics?.singularValues || []
  const diagonalizable = diagnostics?.diagonalizable
  const normal = diagnostics?.normal
  const orthonormal = diagnostics?.orthonormal
  const orthogonal = diagnostics?.orthogonal
  const unitary = diagnostics?.unitary
  const definiteness = diagnostics?.definiteness
  const spectralClass = diagnostics?.spectralClass || diagnostics?.spectral
  const characteristic = formatPolynomial(diagnostics?.characteristicPolynomial)

  return (
    <MatrixAnalysisLayout
      matrixString={matrixString}
      diagnostics={diagnostics}
      activeTab="spectral"
      title="Spectral Analysis"
      subtitle="Eigendecomposition and spectral characteristic identification"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Spectral Analysis' }]} />}
      actions={<button className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 hover:bg-slate-50 rounded-xl text-sm font-bold transition-all text-slate-700"><span className="material-symbols-outlined text-[20px]">description</span>Export LaTeX</button>}
    >
      <div className="p-8 flex-1 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-6">
            {/* Eigenvalues Card */}
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
              <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex justify-between items-center">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Eigenvalues</h4>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: overallSeverity.color }} title={overallSeverity.issues?.join('; ')}>{overallSeverity.label}</span>
                  <span className="text-[10px] bg-primary/10 text-primary px-2 py-0.5 rounded font-bold"><Latex tex={'\\sigma(A)'} /></span>
                </div>
              </div>
              <div className="p-5 space-y-4">
                {eigenvalues.length === 0 && <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Eigenvalues unavailable.</div>}

                {eigenvalues.length > 0 && (
                  <div className="flex flex-wrap gap-2">{eigenvalues.map((lambda, i) => (<span key={`all-${i}`} className="text-[11px] text-slate-600 bg-slate-100 px-2 py-0.5 rounded"><Latex tex={`\\lambda_{${i + 1}} = ${formatComplex(lambda,3)}`} /></span>))}</div>
                )}

                {/* unique eigenvalues */}
                {uniqueEigenvalues.map((lambda, idx) => {
                  const origIdx = findOriginalIndex(lambda)
                  const evSeverity = computePerEigenvalueSeverity(lambda, origIdx >= 0 ? origIdx : idx, diagnostics)
                  // find matching per-eigen info (if available)
                  let info = null
                  if (perEigenInfo && perEigenInfo.length) {
                    for (const it of perEigenInfo) {
                      const v = it?.eigenvalue
                      if (!v) continue
                      if (Math.hypot((v.real ?? 0) - (lambda.real ?? 0), (v.imag ?? 0) - (lambda.imag ?? 0)) <= tol) { info = it; break }
                    }
                  }
                  let algVal = info?.algebraicMultiplicity ?? null
                  let geoVal = info?.geometricMultiplicity ?? null
                  if ((algVal === null || geoVal === null) && eigenvalues.length) {
                    for (let j = 0; j < eigenvalues.length; j++) {
                      const evOrig = eigenvalues[j]
                      if (Math.hypot((evOrig?.real ?? 0) - (lambda?.real ?? 0), (evOrig?.imag ?? 0) - (lambda?.imag ?? 0)) <= tol) {
                        algVal = algVal ?? alg[j]
                        geoVal = geoVal ?? geo[j]
                        break
                      }
                    }
                  }
                  return (
                    <div key={`unique-${idx}`} className="flex items-center justify-between group">
                      <div className="flex items-center gap-4"><span className="math-font text-lg font-bold" style={{ color: evSeverity.color }}><Latex tex={`\\lambda = ${formatComplex(lambda, 3)}`} /></span></div>
                      <div className="flex gap-2">
                        <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: evSeverity.color }} title={evSeverity.issues?.join('; ')}>{evSeverity.label}</span>
                      </div>
                    </div>
                  )
                })}

                {perEigenInfo.length === 0 && <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Per-eigenvalue information unavailable.</div>}

                {uniqueEigenvalues.map((lambda, idx) => {
                  // For each unique eigenvalue, find the perEigenInfo (if present) and render one eigenbasis panel
                  let info = null
                  if (perEigenInfo && perEigenInfo.length) {
                    for (const it of perEigenInfo) {
                      const v = it?.eigenvalue
                      if (!v) continue
                      if (Math.hypot((v.real ?? 0) - (lambda.real ?? 0), (v.imag ?? 0) - (lambda.imag ?? 0)) <= tol) { info = it; break }
                    }
                  }
                  const rep = info?.representativeEigenvector || null
                  // Use eigenspace (null space of A - λI), not eigenbasis from V matrix
                  const eigenspaceVectors = info?.eigenspace?.vectors || null
                  const eigenbasisDim = info?.dimension ?? info?.geometricMultiplicity ?? (eigenspaceVectors ? eigenspaceVectors.length : 0)
                  // Transpose eigenspace vectors to display as basis (column vectors)
                  const eigenbasisAsColumns = eigenspaceVectors ? (() => {
                    if (eigenspaceVectors.length === 0) return null
                    const numRows = eigenspaceVectors[0].length
                    const numCols = eigenspaceVectors.length
                    const result = []
                    for (let i = 0; i < numRows; i++) {
                      const row = []
                      for (let j = 0; j < numCols; j++) {
                        row.push(eigenspaceVectors[j][i])
                      }
                      result.push(row)
                    }
                    return result
                  })() : null
                  const origIdx = lambda ? findOriginalIndex(lambda) : -1
                  const evSeverity = lambda ? computePerEigenvalueSeverity(lambda, origIdx >= 0 ? origIdx : idx, diagnostics) : { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }
                  const algMultiplicity = info?.algebraicMultiplicity ?? algVal
                  const geoMultiplicity = info?.dimension ?? info?.geometricMultiplicity ?? geoVal
                  const multiplicityMismatch = algMultiplicity !== null && geoMultiplicity !== null && algMultiplicity !== geoMultiplicity
                  const multiplicityColor = multiplicityMismatch ? SEVERITY_COLORS.severe : SEVERITY_COLORS.safe
                  return (
                    <div key={`basis-${idx}`} className="mt-2 p-3 border-l-4 rounded-r" style={{ borderColor: evSeverity.color, backgroundColor: `${evSeverity.color}08` }}>
                      <div className="flex items-center justify-between mb-2"><div className="text-xs text-slate-500">Eigenspace for {lambda ? <Latex tex={`\\lambda = ${formatComplex(lambda,3)}`} /> : <span>λ</span>}</div><span className="text-[9px] px-1.5 py-0.5 rounded font-bold text-white" style={{ backgroundColor: evSeverity.color }} title={evSeverity.issues?.join('; ')}>{evSeverity.label}</span></div>
                      <div className="flex gap-4 items-start">
                        <div className="flex-1"><div className="text-[10px] text-slate-400 mb-1">Eigenspace (dim: {eigenbasisDim})</div>{eigenbasisAsColumns ? <div style={multiplicityMismatch ? { color: evSeverity.color } : {}}><MatrixDisplay data={eigenbasisAsColumns} minCellWidth={28} gap={8} className="text-sm" cellClassName="text-center" /></div> : <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>—</div>}</div>
                        <div className="flex-1"><div className="text-[10px] text-slate-400 mb-1">Representative Eigenvector</div>{rep ? <div style={multiplicityMismatch ? { color: evSeverity.color } : {}}><MatrixDisplay data={rep.map(v => [v])} minCellWidth={28} gap={8} className="text-sm" cellClassName="text-center" /></div> : <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>—</div>}</div>
                        <div className="flex-1"><div className="text-[10px] text-slate-400 mb-1">Algebraic / Dimension</div><div className="text-sm"><div>alg: <span style={algMultiplicity === undefined || algMultiplicity === null ? { color: SEVERITY_COLORS.critical } : { color: multiplicityColor }}>{algMultiplicity ?? '—'}</span></div><div>dim: <span style={geoMultiplicity === undefined || geoMultiplicity === null ? { color: SEVERITY_COLORS.critical } : { color: multiplicityColor }}>{geoMultiplicity ?? '—'}</span></div></div></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Characteristic Polynomial */}
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm"><div className="bg-slate-50 px-5 py-3 border-b border-slate-200"><h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Characteristic Polynomial</h4></div><div className="p-8 text-center"><p className="math-font text-xl text-slate-800"><Latex tex={characteristic} /></p></div></div>

            {/* Singular Values */}
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm"><div className="bg-slate-50 px-5 py-3 border-b border-slate-200"><h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Singular Values</h4></div><div className="p-4">{(!singularValues || singularValues.length === 0) ? (<div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Singular values unavailable.</div>) : (<div className="flex gap-2 flex-wrap">{singularValues.map((s, i) => (<span key={i} className="text-xs text-slate-700 bg-slate-100 px-2 py-1 rounded">{formatNumber(s, 4)}</span>))}</div>)}</div></div>
          </div>

          {/* Right column */}
          <div className="bg-slate-50 rounded-xl border border-slate-100 p-6 flex flex-col h-full">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Eigenvectors Matrix (V)</h4>
                {(() => {
                  const sumAlg = alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
                  const matrixCols = diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
                  return sumAlg !== matrixCols && sumAlg > 0 && matrixCols > 0 ? (
                    <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
                  ) : null
                })()}
              </div>
              <button className="text-primary hover:text-primary/80 transition-colors">
              <span className="material-symbols-outlined text-[20px]">content_copy</span></button></div>
            <div className="flex-1 flex items-center justify-center"><div className="flex items-stretch gap-2"><div className="matrix-bracket-left"></div><MatrixDisplay data={eigenvectors} minCellWidth={40} gap={16} className="math-font text-sm font-medium py-4 px-4" cellClassName="text-center" highlightDiagonal style={criticalArtifacts.eigenvectorMatrix ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 8 } : {}} /><div className="matrix-bracket-right"></div></div></div>
            <div className="mt-6 space-y-4"><div className="h-[1px] bg-slate-200"></div><div><h4 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-tighter">Quick Summary</h4><ul className="space-y-4"><li className="flex items-start gap-3"><span className="material-symbols-outlined text-primary text-sm mt-1">check_circle</span><div><p className="text-sm font-bold"><Latex tex={'\\text{Spectral Radius} \\ (\\rho)'} /></p><p className="text-lg font-bold math-font text-primary">{formatNumber(diagnostics?.spectralRadius, 4)}</p><p className="text-xs text-slate-500">Maximum absolute eigenvalue.</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${diagnostics?.defectivity ? 'text-rose-500' : 'text-emerald-600'}`}>{diagnostics?.defectivity ? 'error' : 'check_circle'}</span><div><p className="text-sm font-bold">{diagnostics?.defectivity ? 'Defective' : 'Non-defective'}</p><p className="text-xs text-slate-500">{diagnostics?.defectivity ? 'Eigenspace dimension is smaller than matrix size.' : 'Eigenspace dimension matches matrix size.'}</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${diagonalizable ? 'text-emerald-600' : 'text-rose-500'}`}>{diagonalizable ? 'check_circle' : 'error'}</span><div><p className="text-sm font-bold">{diagonalizable ? 'Diagonalizable' : 'Not diagonalizable'}</p><p className="text-xs text-slate-500">{diagonalizable ? 'Matrix admits a full eigenbasis.' : 'Insufficient linearly independent eigenvectors.'}</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${normal ? 'text-emerald-600' : 'text-amber-500'}`}>{normal ? 'check_circle' : 'warning'}</span><div><p className="text-sm font-bold">{normal ? 'Normal' : 'Non-normal'}</p><p className="text-xs text-slate-500">{normal ? 'Commutes with its adjoint.' : 'Does not commute with its adjoint.'}</p></div></li><li className="flex items-start gap-3"><div><p className="text-sm font-bold">Orthonormal: {orthonormal ? 'Yes' : 'No'}</p><p className="text-sm font-bold">Orthogonal: {orthogonal ? 'Yes' : 'No'}</p><p className="text-sm font-bold">Unitary: {unitary ? 'Yes' : 'No'}</p></div></li><li className="flex items-start gap-3"><div><p className="text-sm font-bold">Definiteness: <span style={definiteness === undefined || definiteness === null ? { color: SEVERITY_COLORS.critical } : {}}>{definiteness ?? 'Unknown'}</span></p><p className="text-sm font-bold">Spectral Class: <span style={spectralClass === undefined || spectralClass === null ? { color: SEVERITY_COLORS.critical } : {}}>{spectralClass ?? 'Unknown'}</span></p></div></li></ul></div></div>
          </div>
        </div>
      </div>

      {/* Eigendecomposition card */}
      <div className="p-8">
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Eigendecomposition</h4>
              {(() => {
                const sumAlg = alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
                const matrixCols = diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
                return sumAlg !== matrixCols && sumAlg > 0 && matrixCols > 0 ? (
                  <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
                ) : null
              })()}
            </div>
            <div className="text-xs" style={{ color: diagStatus ? undefined : SEVERITY_COLORS.critical }}>{diagStatus ?? 'Unavailable'}</div>
          </div>
          <div className="p-6 grid grid-cols-3 gap-6 items-start"><div><div className="text-[10px] text-slate-400 mb-2">P (eigenvector matrix)</div>{diagP ? (<div className="flex items-stretch gap-2"><div className="matrix-bracket-left"></div><MatrixDisplay data={diagP.data || diagP} minCellWidth={40} gap={12} className="math-font text-sm" cellClassName="text-center" style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}} /><div className="matrix-bracket-right"></div></div>) : (<div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Eigenvector matrix unavailable.</div>)}</div><div><div className="text-[10px] text-slate-400 mb-2">D (diagonal eigenvalue matrix)</div>{diagD ? (<div className="flex items-stretch gap-2"><div className="matrix-bracket-left"></div><MatrixDisplay data={diagD.data || diagD} minCellWidth={40} gap={12} className="math-font text-sm" cellClassName="text-center" style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}} /><div className="matrix-bracket-right"></div></div>) : (<div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Diagonal matrix unavailable.</div>)}</div><div><div className="text-[10px] text-slate-400 mb-2"><Latex tex={'P^{-1}'} /> (inverse eigenvector matrix)</div>{diagPInverse ? (<div className="flex items-stretch gap-2"><div className="matrix-bracket-left"></div><MatrixDisplay data={diagPInverse.data || diagPInverse} minCellWidth={40} gap={12} className="math-font text-sm" cellClassName="text-center" style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}} /><div className="matrix-bracket-right"></div></div>) : (<div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Inverse matrix unavailable.</div>)}</div></div>
        </div>
      </div>

      {/* Global Eigenspace card */}
      <div className="p-8">
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex items-center gap-2">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Eigenbasis</h4>
            {(() => {
              const sumAlg = alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
              const matrixCols = diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
              return sumAlg !== matrixCols && sumAlg > 0 && matrixCols > 0 ? (
                <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
              ) : null
            })()}
          </div>
          <div className="p-6">{(!eigenspace || eigenspace.length === 0) ? (<div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Eigenspace unavailable.</div>) : (<div className="flex items-stretch gap-2"><div className="matrix-bracket-left"></div><MatrixDisplay data={eigenspace} minCellWidth={28} gap={12} className="text-sm" cellClassName="text-center" /><div className="matrix-bracket-right"></div></div>)}</div>
        </div>
      </div>
      <div className="bg-slate-50 border-t border-slate-200 px-8 py-4 flex items-center justify-between"><span className="text-xs text-slate-400 italic">Solved using high-precision JLA subroutines.</span><div className="flex gap-6"><a className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5" href={`/matrix=${encodeURIComponent(matrixString)}/report`} onClick={(e) => { e.preventDefault(); import('../utils/navigation').then(m => m.navigate(`/matrix=${encodeURIComponent(matrixString)}/report`)) }}><span className="material-symbols-outlined text-[18px]">visibility</span>Full Report</a><button className="text-xs font-bold text-primary hover:text-primary/80 flex items-center gap-1.5"><span className="material-symbols-outlined text-[18px]">download</span>JSON Data</button></div></div>
    </MatrixAnalysisLayout>
  )
}



