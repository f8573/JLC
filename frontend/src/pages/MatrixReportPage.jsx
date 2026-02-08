import React, { useState } from 'react'
import MatrixDisplay from '../components/matrix/MatrixDisplay'
import MatrixLatex from '../components/matrix/MatrixLatex'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { usePrecisionUpdate } from '../hooks/usePrecisionUpdate'
import { formatComplex, formatDimension, formatNumber, formatDefiniteness } from '../utils/format'
import Latex from '../components/ui/Latex'
import Logo from '../components/ui/Logo'
import SpectrumMap from '../components/results/SpectrumMap'

/**
 * Format the report timestamp in the local browser locale.
 *
 * @returns {string}
 */
function formatDate() {
  return new Date().toLocaleString()
}

/**
 * Convert polynomial coefficients to a readable report string.
 *
 * @param {Array<{real:number, imag:number}>} coeffs
 * @returns {string}
 */
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
    if (power > 0 && (coeffStr === '1' || coeffStr === '1.0')) {
      coeffStr = ''
    }
    if (power > 0 && (coeffStr === '-1' || coeffStr === '-1.0')) {
      coeffStr = '-'
    }
    const lambdaPart = power === 0 ? '' : power === 1 ? '\\lambda' : `\\lambda^${power}`
    terms.push(`${coeffStr}${lambdaPart}`.trim())
  })
  return terms.length ? `p(\\lambda) = ${terms.join(' + ').replace(/\+ -/g, '- ')}` : '—'
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
      <div className="relative w-11/12 md:w-3/4 lg:w-2/3 max-h-[80vh] bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl shadow-lg overflow-hidden transition-colors duration-300">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 dark:border-slate-700">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">JSON Data</h3>
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
          <pre className="whitespace-pre-wrap text-sm font-mono text-slate-800 dark:text-slate-200">
            {JSON.stringify({ matrixString, diagnostics }, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}

/**
 * Print-ready matrix analysis report view.
 *
 * @param {Object} props
 * @param {string} props.matrixString - Serialized matrix payload from the URL.
 */
export default function MatrixReportPage({ matrixString }) {
  // Subscribe to precision changes
  usePrecisionUpdate()
  
  const { diagnostics } = useDiagnostics(matrixString)
  const [jsonModalOpen, setJsonModalOpen] = useState(false)

  const rows = diagnostics?.rows
  const cols = diagnostics?.columns
  const dimensionLabel = formatDimension(rows, cols)
  const matrixData = diagnostics?.matrixData
  const eigenvalues = diagnostics?.eigenvalues || []
  const alg = diagnostics?.algebraicMultiplicity || []
  const eigenvectors = diagnostics?.eigenvectors || null
  const luL = diagnostics?.lu?.l
  const luU = diagnostics?.lu?.u
  const svdSigma = diagnostics?.svd?.sigma
  const characteristic = formatPolynomial(diagnostics?.characteristicPolynomial)

  const determinant = formatNumber(diagnostics?.determinant, 4)
  const rank = diagnostics?.rank ?? '—'
  const symmetry = diagnostics?.symmetric ? 'Symmetric' : 'Asymmetric'
  const diagD = diagnostics?.diagonalization?.d || diagnostics?.d
  const diagP = diagnostics?.diagonalization?.p || diagnostics?.p

  const matrixToDisplay = (matrix) => {
    if (!matrix) return null
    const data = matrix.data || matrix
    if (!Array.isArray(data)) return null
    return matrix
  }

  const eigenvectorsDisplay = matrixToDisplay(eigenvectors)

  const handleBackToBasic = () => {
    window.location.href = `/matrix=${encodeURIComponent(matrixString)}/basic`
  }

  return (
    <div className="bg-slate-50 dark:bg-slate-900 font-display text-slate-900 dark:text-slate-100 h-screen overflow-hidden flex flex-col transition-colors duration-300">
      <header className="flex items-center justify-between border-b border-solid border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-3 sticky top-0 z-50 no-print transition-colors duration-300">
        <div className="flex items-center gap-8">
          <button 
            onClick={handleBackToBasic}
            className="flex items-center gap-1 text-slate-500 hover:text-primary transition-colors"
            title="Back to Analysis"
          >
            <span className="material-symbols-outlined text-[20px]">arrow_back</span>
          </button>
          <a href="/" className="flex items-center gap-3">
            <Logo />
            <h2 className="text-lg font-bold leading-tight tracking-tight text-slate-800 dark:text-slate-100">ΛCompute</h2>
          </a>
        </div>
        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-6">
            <a className="text-sm font-semibold text-slate-600 dark:text-slate-300 hover:text-primary" href="/documentation">Documentation</a>
          </nav>
          <div className="h-8 w-[1px] bg-slate-200 dark:bg-slate-700 mx-2"></div>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-bold shadow-md hover:bg-primary/90 transition-all" onClick={() => window.print()}>
            <span className="material-symbols-outlined text-[20px]">print</span>
            Export PDF
          </button>
        </div>
      </header>
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="max-w-[1200px] mx-auto flex gap-8 p-8">
          <main className="flex-1 bg-white dark:bg-slate-800 border border-border-elegant dark:border-slate-700 shadow-sm rounded-xl p-12 report-container min-h-[1500px] transition-colors duration-300 dark:[&_.bg-white]:bg-slate-800 dark:[&_.bg-slate-50]:bg-slate-800/60 dark:[&_.bg-slate-50\\/50]:bg-slate-800/60 dark:[&_.border-slate-100]:border-slate-700 dark:[&_.border-slate-200]:border-slate-700 dark:[&_.text-slate-900]:text-slate-100 dark:[&_.text-slate-800]:text-slate-100 dark:[&_.text-slate-700]:text-slate-200 dark:[&_.text-slate-600]:text-slate-300 dark:[&_.text-slate-500]:text-slate-400 dark:[&_.text-slate-400]:text-slate-500">
          <header className="mb-12">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h1 className="text-4xl font-black text-slate-900 dark:text-slate-100 mb-2">Matrix Analysis Report</h1>
                <div className="flex gap-4 text-sm text-slate-400 dark:text-slate-500 font-medium">
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[16px]">fingerprint</span>
                    ID: #MS-99421-A
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[16px]">schedule</span>
                    {formatDate()}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="px-4 py-2 bg-primary-light rounded-lg border border-primary/20">
                  <p className="text-[10px] font-bold text-primary">Global Status</p>
                  <p className="text-lg font-bold text-primary">{diagnostics?.wellConditioned ? 'Well-Conditioned' : 'Needs Review'}</p>
                </div>
              </div>
            </div>
                <div className="grid grid-cols-4 gap-4 p-6 bg-slate-50 dark:bg-slate-700/40 border border-slate-100 dark:border-slate-700 rounded-xl transition-colors duration-300">
              <div>
                <p className="text-[10px] font-bold text-slate-400">Matrix Class</p>
                <p className="font-bold text-slate-800 dark:text-slate-100">{diagnostics?.square ? 'Square' : 'Rectangular'}, {diagnostics?.domain === 'C' ? 'Complex' : 'Real'}</p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400">Determinant</p>
                    <p className="font-bold text-slate-800 dark:text-slate-100 math-text"><Latex tex={`\\det(A) = ${determinant}`} /></p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400">Rank</p>
                <p className="font-bold text-slate-800 dark:text-slate-100">{rank}</p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400">Symmetry</p>
                <p className="font-bold text-slate-800 dark:text-slate-100">{symmetry}</p>
              </div>
            </div>
          </header>

          <section className="report-section" id="input">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">1</span>
              Input Details
            </h2>
            <div className="flex items-center justify-center gap-12 py-4">
              <div className="flex items-center gap-6">
                <span className="math-text text-3xl font-bold">A =</span>
                <div className="matrix-bracket math-text text-2xl tracking-widest leading-relaxed">
                  <MatrixDisplay
                    data={matrixData}
                    minCellWidth={60}
                    gap={8}
                    className="grid"
                    cellClassName="text-center"
                  />
                </div>
              </div>
              <div className="w-48 text-sm text-slate-500 italic space-y-2 border-l border-slate-100 pl-6">
                <p>• Size: {dimensionLabel}</p>
                <p>• Entries: Numeric</p>
                <p>• Norm (inf): {formatNumber(diagnostics?.normInf, 4)}</p>
              </div>
            </div>
          </section>

          <section className="report-section" id="properties">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">2</span>
              Basic Properties
            </h2>
            <div className="grid grid-cols-2 gap-x-12 gap-y-8">
              <div className="flex justify-between items-center border-b border-slate-50 pb-4">
                <span className="math-text text-slate-600">Rank(A)</span>
                <span className="font-bold text-lg">{rank}</span>
              </div>
              <div className="flex justify-between items-center border-b border-slate-50 pb-4">
                <span className="math-text text-slate-600">det(A)</span>
                <span className="font-bold text-lg">{determinant}</span>
              </div>
              <div className="flex justify-between items-center border-b border-slate-50 pb-4">
                <span className="math-text text-slate-600">tr(A)</span>
                <span className="font-bold text-lg">{formatNumber(diagnostics?.trace, 4)}</span>
              </div>
              <div className="flex justify-between items-center border-b border-slate-50 pb-4">
                <span className="math-text text-slate-600">Cond(A)</span>
                <span className="font-bold text-lg text-primary"><Latex tex={`\\kappa(A) = ${formatNumber(diagnostics?.conditionNumber, 4)}`} /></span>
              </div>
            </div>
          </section>
          
          <section className="report-section" id="echelon">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">3</span>
              Echelon Forms &amp; Attributes
            </h2>
            <div className="grid grid-cols-2 gap-8">
              <div>
                <h3 className="text-sm font-bold text-slate-700 mb-4">Echelon Forms</h3>
                <div className="space-y-4 bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <div>
                    <p className="text-xs text-slate-400 mb-2">Reduced Row Echelon Form (RREF)</p>
                    {diagnostics?.rrefMatrix?.data ? (
                      <MatrixLatex data={diagnostics.rrefMatrix} className="math-text text-sm" precision={2} />
                    ) : (
                      <p className="text-sm text-slate-400">Unavailable</p>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 mb-2">Row Echelon (Upper Triangular Form)</p>
                    {diagnostics?.rowEchelonMatrix?.data ? (
                      <MatrixLatex data={diagnostics.rowEchelonMatrix} className="math-text text-sm" precision={2} />
                    ) : (
                      <p className="text-sm text-slate-400">Unavailable</p>
                    )}
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold text-slate-700 mb-4">Attributes</h3>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 space-y-3">
                  <div>
                    <p className="text-xs text-slate-400 mb-2">True Attributes</p>
                    <div className="flex flex-wrap gap-2">
                      {[
                        ['Invertible', diagnostics?.invertible],
                        ['Singular', diagnostics?.singular],
                        ['Full Rank', diagnostics?.fullRank],
                        ['Row Echelon', diagnostics?.rowEchelon],
                        ['Symmetric', diagnostics?.symmetric],
                        ['Orthogonal', diagnostics?.orthogonal],
                        ['Hessenberg', diagnostics?.hessenberg],
                        ['Identity', diagnostics?.isIdentity]
                      ].filter(([label, val]) => !!val).map(([label]) => (
                        <span key={label} className="text-[12px] px-2 py-1 bg-emerald-50 text-emerald-700 rounded font-semibold">{label}</span>
                      ))}
                      {(!diagnostics || Object.keys(diagnostics).length === 0) && <div className="text-sm text-slate-400">No attributes available</div>}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs text-slate-400 mb-2">Numeric Attributes</p>
                    <div className="grid grid-cols-2 gap-2 text-sm text-slate-700">
                      <div className="flex justify-between"><span className="text-slate-400">Density</span><span className="font-bold">{diagnostics?.density != null ? formatNumber(diagnostics.density, 3) : '—'}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Condition #</span><span className="font-bold">{diagnostics?.conditionNumber != null ? formatNumber(diagnostics.conditionNumber, 3) : '—'}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Norm (1)</span><span className="font-bold">{diagnostics?.norm1 != null ? formatNumber(diagnostics.norm1, 4) : '—'}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Norm (∞)</span><span className="font-bold">{diagnostics?.normInf != null ? formatNumber(diagnostics.normInf, 4) : '—'}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Frobenius Norm</span><span className="font-bold">{diagnostics?.frobeniusNorm != null ? formatNumber(diagnostics.frobeniusNorm, 4) : '—'}</span></div>
                      <div className="flex justify-between"><span className="text-slate-400">Spectral Radius</span><span className="font-bold">{diagnostics?.spectralRadius != null ? formatNumber(diagnostics.spectralRadius, 4) : '—'}</span></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="report-section" id="spectral">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">4</span>
              Spectral Analysis
            </h2>
            <div className="mb-10">
              <h3 className="text-xs font-bold text-slate-400 mb-4">Eigenvalues &amp; Distribution</h3>
              <div className="flex gap-8 items-start">
                <div className="flex-1">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-slate-500 text-[10px]">
                      <tr>
                        <th className="py-3 px-4 text-left font-bold">Index</th>
                        <th className="py-3 px-4 text-left font-bold"><Latex tex="\lambda" /></th>
                        <th className="py-3 px-4 text-left font-bold">Algebraic Mult.</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {eigenvalues.length === 0 && (
                        <tr>
                          <td className="py-3 px-4 text-slate-400" colSpan="3">Eigenvalues unavailable.</td>
                        </tr>
                      )}
                      {eigenvalues.map((lambda, idx) => (
                        <tr key={idx}>
                          <td className="py-3 px-4 font-medium text-slate-400"><Latex tex={`\\lambda_{${idx + 1}}`} /></td>
                          <td className="py-3 px-4 math-text font-bold"><Latex tex={formatComplex(lambda, 3)} /></td>
                          <td className="py-3 px-4">{alg[idx] ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="w-48 h-32 p-2">
                  <div className="text-[9px] font-bold text-primary p-1">Spectrum Map</div>
                  <div className="mt-1">
                    {/* New interactive spectrum map component */}
                    <React.Suspense fallback={<div className="text-xs text-slate-400">Loading map...</div>}>
                      <SpectrumMap compact={true} eigenvalues={eigenvalues} spectralRadius={diagnostics?.spectralRadius || 0} width={128} height={128} />
                    </React.Suspense>
                  </div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-12">
              <div>
                <h3 className="text-xs font-bold text-slate-400 mb-4">Characteristic Polynomial</h3>
                <div className="p-6 bg-surface-accent border border-border-elegant rounded-xl text-center">
                  <p className="math-text text-xl"><Latex tex={characteristic} /></p>
                </div>
              </div>
              <div>
                <h3 className="text-xs font-bold text-slate-400 mb-4">Eigenvector Matrix (V)</h3>
                <div className="flex justify-center py-2">
                  <MatrixLatex matrix={eigenvectors} className="math-text text-sm" precision={3} />
                </div>
              </div>
            </div>
          </section>

          <section className="report-section" id="decompositions">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">5</span>
              Matrix Decompositions
            </h2>
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">LU Decomposition</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={luL} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={luU} precision={2} />
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">SVD (USV*)</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.svd?.u} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={svdSigma} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.svd?.v} precision={2} />
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-400 italic">Truncated for display. Full precision in JSON export.</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">QR Decomposition</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.qr?.q} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.qr?.r} precision={2} />
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">Cholesky / Polar / Hessenberg</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.cholesky?.l} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.polar?.u} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.hessenbergDecomposition?.h} precision={2} />
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">Schur / Spectral</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.schurDecomposition?.u} precision={2} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={diagnostics?.schurDecomposition?.t} precision={2} />
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <h4 className="text-sm font-bold text-slate-700">Eigendecomposition</h4>
                  <div className="flex items-center gap-4">
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={matrixToDisplay(diagP)} precision={3} />
                    </div>
                    <div className="math-text text-[11px] scale-90">
                      <MatrixLatex data={matrixToDisplay(diagD)} precision={3} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="report-section border-b-0" id="structure">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">6</span>
              Structural Analysis
            </h2>
            <div className="grid grid-cols-3 gap-6">
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 mb-2">Definiteness</p>
                <p className="font-bold text-slate-700">{formatDefiniteness(diagnostics?.definiteness)}</p>
                <p className="text-[10px] text-slate-500 mt-1">Eigenvalues determine sign structure.</p>
              </div>
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 mb-2">Orthogonality</p>
                <p className="font-bold text-slate-700">{diagnostics?.orthogonal ? 'Orthogonal' : 'Non-Orthogonal'}</p>
                <p className="text-[10px] text-slate-500 mt-1"><Latex tex="A^T A \approx I" /></p>
              </div>
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 mb-2">Symmetry Type</p>
                <p className="font-bold text-slate-700">{symmetry}</p>
                <p className="text-[10px] text-slate-500 mt-1"><Latex tex="A = A^T" /></p>
              </div>
            </div>
          </section>
          <footer className="mt-12 pt-8 border-t border-slate-100 dark:border-slate-700 flex justify-between items-center text-[10px] text-slate-400 dark:text-slate-500 font-bold">
            <span>JLA HPC Engine v1.0.0</span>
            <span>Page 1 of 1</span>
            <span>Confidential Analysis Report</span>
          </footer>
          </main>
          <aside className="w-64 no-print h-fit sticky top-24">
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 transition-colors duration-300">
            <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">list_alt</span>
              Contents
            </h4>
            <nav className="space-y-1">
              <a className="sticky-toc-link active" href="#input">1. Input Details</a>
              <a className="sticky-toc-link" href="#properties">2. Basic Properties</a>
              <a className="sticky-toc-link" href="#echelon">3. Echelon Forms &amp; Attributes</a>
              <a className="sticky-toc-link" href="#spectral">4. Spectral Analysis</a>
              <a className="sticky-toc-link" href="#decompositions">5. Decompositions</a>
              <a className="sticky-toc-link" href="#structure">6. Structure</a>
            </nav>
            <div className="mt-8 pt-6 border-t border-slate-100 dark:border-slate-700">
              <button 
                onClick={() => setJsonModalOpen(true)}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-50 dark:bg-slate-700 text-slate-600 dark:text-slate-200 rounded-lg text-xs font-bold hover:bg-slate-100 dark:hover:bg-slate-600 transition-all border border-slate-200 dark:border-slate-600 mb-2"
              >
                <span className="material-symbols-outlined text-[18px]">download</span>
                Raw JSON Data
              </button>
            </div>
          </div>
          <div className="mt-6 p-4 rounded-xl bg-primary/5 dark:bg-primary/10 border border-primary/10 dark:border-primary/20 transition-colors duration-300">
            <p className="text-[10px] font-bold text-primary mb-1">Analysis Mode</p>
            <p className="text-xs text-slate-600 dark:text-slate-300">High-precision double float (64-bit) JLA backend.</p>
          </div>
          </aside>
        </div>
      </div>
      <JsonModal 
        open={jsonModalOpen} 
        onClose={() => setJsonModalOpen(false)} 
        matrixString={matrixString} 
        diagnostics={diagnostics} 
      />
    </div>
  )
}
