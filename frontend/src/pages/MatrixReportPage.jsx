import React from 'react'
import MatrixDisplay from '../components/matrix/MatrixDisplay'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { formatComplex, formatDimension, formatNumber } from '../utils/format'
import Latex from '../components/ui/Latex'

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
    const lambdaPart = power === 0 ? '' : power === 1 ? '?' : `?^${power}`
    terms.push(`${coeffStr}${lambdaPart}`.trim())
  })
  return terms.length ? `p(?) = ${terms.join(' + ').replace(/\+ -/g, '- ')}` : '—'
}

/**
 * Print-ready matrix analysis report view.
 *
 * @param {Object} props
 * @param {string} props.matrixString - Serialized matrix payload from the URL.
 */
export default function MatrixReportPage({ matrixString }) {
  const { diagnostics } = useDiagnostics(matrixString)

  const rows = diagnostics?.rows
  const cols = diagnostics?.columns
  const dimensionLabel = formatDimension(rows, cols)
  const matrixData = diagnostics?.matrixData
  const eigenvalues = diagnostics?.eigenvalues || []
  const alg = diagnostics?.algebraicMultiplicity || []
  const eigenvectors = diagnostics?.eigenvectors?.data
  const luL = diagnostics?.lu?.l?.data
  const luU = diagnostics?.lu?.u?.data
  const svdSigma = diagnostics?.svd?.sigma?.data
  const characteristic = formatPolynomial(diagnostics?.characteristicPolynomial)

  const determinant = formatNumber(diagnostics?.determinant, 4)
  const rank = diagnostics?.rank ?? '—'
  const symmetry = diagnostics?.symmetric ? 'Symmetric' : 'Asymmetric'

  return (
    <div className="bg-slate-50 font-display text-slate-900 min-h-screen">
      <header className="flex items-center justify-between border-b border-solid border-slate-200 bg-white px-6 py-3 sticky top-0 z-50 no-print">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="size-9 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/20">
              <span className="text-xl font-bold italic">?</span>
            </div>
            <h2 className="text-lg font-bold leading-tight tracking-tight text-slate-800">MatrixSolve AI</h2>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-6">
            <a className="text-sm font-semibold text-slate-600 hover:text-primary" href="#">Documentation</a>
          </nav>
          <div className="h-8 w-[1px] bg-slate-200 mx-2"></div>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg text-sm font-bold shadow-md hover:bg-primary/90 transition-all" onClick={() => window.print()}>
            <span className="material-symbols-outlined text-[20px]">print</span>
            Export PDF
          </button>
        </div>
      </header>
      <div className="max-w-[1200px] mx-auto flex gap-8 p-8">
        <main className="flex-1 bg-white border border-border-elegant shadow-sm rounded-xl p-12 report-container min-h-[1500px]">
          <header className="mb-12">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h1 className="text-4xl font-black text-slate-900 mb-2">Matrix Analysis Report</h1>
                <div className="flex gap-4 text-sm text-slate-400 font-medium">
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
                  <p className="text-[10px] font-bold text-primary uppercase tracking-widest">Global Status</p>
                  <p className="text-lg font-bold text-primary">{diagnostics?.wellConditioned ? 'Well-Conditioned' : 'Needs Review'}</p>
                </div>
              </div>
            </div>
                <div className="grid grid-cols-4 gap-4 p-6 bg-slate-50 border border-slate-100 rounded-xl">
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Matrix Class</p>
                <p className="font-bold text-slate-800">{diagnostics?.square ? 'Square' : 'Rectangular'}, {diagnostics?.domain === 'C' ? 'Complex' : 'Real'}</p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Determinant</p>
                    <p className="font-bold text-slate-800 math-text"><Latex tex={`\\det(A) = ${determinant}`} /></p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Rank</p>
                <p className="font-bold text-slate-800">{rank}</p>
              </div>
              <div>
                <p className="text-[10px] font-bold text-slate-400 uppercase">Symmetry</p>
                <p className="font-bold text-slate-800">{symmetry}</p>
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

          <section className="report-section" id="spectral">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">3</span>
              Spectral Analysis
            </h2>
            <div className="mb-10">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Eigenvalues &amp; Distribution</h3>
              <div className="flex gap-8 items-start">
                <div className="flex-1">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] tracking-wider">
                      <tr>
                        <th className="py-3 px-4 text-left font-bold">Index</th>
                        <th className="py-3 px-4 text-left font-bold">Value (?)</th>
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
                          <td className="py-3 px-4 font-medium text-slate-400">?{idx + 1}</td>
                          <td className="py-3 px-4 math-text font-bold"><Latex tex={formatComplex(lambda, 3)} /></td>
                          <td className="py-3 px-4">{alg[idx] ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="w-48 h-32 bg-primary-light/30 rounded border border-primary/10 p-2 flex flex-col justify-end gap-1">
                  <p className="text-[9px] font-bold text-primary absolute p-1 uppercase">Spectrum Map</p>
                  <div className="flex items-end justify-center gap-2 h-full">
                    <div className="w-6 bg-primary/40 h-[90%] rounded-t-sm"></div>
                    <div className="w-6 bg-primary/20 h-[40%] rounded-t-sm"></div>
                    <div className="w-6 bg-primary/20 h-[40%] rounded-t-sm"></div>
                  </div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-12">
              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Characteristic Polynomial</h3>
                <div className="p-6 bg-surface-accent border border-border-elegant rounded-xl text-center">
                  <p className="math-text text-xl"><Latex tex={characteristic} /></p>
                </div>
              </div>
              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Eigenvector Matrix (V)</h3>
                <div className="flex justify-center py-2">
                  <div className="matrix-bracket math-text text-sm">
                    <MatrixDisplay
                      data={eigenvectors}
                      minCellWidth={40}
                      gap={8}
                      className="grid"
                      cellClassName="text-center"
                    />
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="report-section" id="decompositions">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">4</span>
              Matrix Decompositions
            </h2>
            <div className="grid grid-cols-2 gap-12">
              <div className="space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">LU Decomposition</h3>
                  <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded text-slate-500">Crout's Method</span>
                </div>
                <div className="flex flex-col gap-6 items-center">
                  <div className="flex items-center gap-3">
                    <span className="math-text text-sm">L =</span>
                    <div className="matrix-bracket math-text text-[11px] scale-90">
                      <MatrixDisplay
                        data={luL}
                        minCellWidth={30}
                        gap={8}
                        className="grid"
                        cellClassName="text-center"
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="math-text text-sm">U =</span>
                    <div className="matrix-bracket math-text text-[11px] scale-90">
                      <MatrixDisplay
                        data={luU}
                        minCellWidth={30}
                        gap={8}
                        className="grid"
                        cellClassName="text-center"
                      />
                    </div>
                  </div>
                </div>
              </div>
              <div className="space-y-6">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">SVD (USV*)</h3>
                  <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded text-slate-500">Numerical Robust</span>
                </div>
                <div className="flex flex-col gap-4 items-center">
                  <div className="flex items-center gap-2">
                    <span className="math-text text-[10px]">S = diag</span>
                    <div className="matrix-bracket math-text text-[11px] scale-75">
                      <MatrixDisplay
                        data={svdSigma}
                        minCellWidth={30}
                        gap={8}
                        className="grid"
                        cellClassName="text-center"
                      />
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-400 italic">Truncated for display. Full precision in JSON export.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="report-section border-b-0" id="structure">
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3">
              <span className="size-8 bg-slate-100 rounded flex items-center justify-center text-slate-400 text-sm">5</span>
              Structural Analysis
            </h2>
            <div className="grid grid-cols-3 gap-6">
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Definiteness</p>
                <p className="font-bold text-slate-700">{diagnostics?.definiteness ?? 'Unknown'}</p>
                <p className="text-[10px] text-slate-500 mt-1">Eigenvalues determine sign structure.</p>
              </div>
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Orthogonality</p>
                <p className="font-bold text-slate-700">{diagnostics?.orthogonal ? 'Orthogonal' : 'Non-Orthogonal'}</p>
                <p className="text-[10px] text-slate-500 mt-1">A^T A ˜ I</p>
              </div>
              <div className="p-5 rounded-xl border border-border-elegant bg-slate-50/50">
                <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Symmetry Type</p>
                <p className="font-bold text-slate-700">{symmetry}</p>
                <p className="text-[10px] text-slate-500 mt-1">A compared with A^T.</p>
              </div>
            </div>
          </section>
          <footer className="mt-12 pt-8 border-t border-slate-100 flex justify-between items-center text-[10px] text-slate-400 font-bold uppercase tracking-widest">
            <span>MatrixSolve AI Engine v4.2.0</span>
            <span>Page 1 of 1</span>
            <span>Confidential Analysis Report</span>
          </footer>
        </main>
        <aside className="w-64 no-print h-fit sticky top-24">
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h4 className="text-xs font-bold text-slate-900 uppercase mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">list_alt</span>
              Contents
            </h4>
            <nav className="space-y-1">
              <a className="sticky-toc-link active" href="#input">1. Input Details</a>
              <a className="sticky-toc-link" href="#properties">2. Basic Properties</a>
              <a className="sticky-toc-link" href="#spectral">3. Spectral Analysis</a>
              <a className="sticky-toc-link" href="#decompositions">4. Decompositions</a>
              <a className="sticky-toc-link" href="#structure">5. Structure</a>
            </nav>
            <div className="mt-8 pt-6 border-t border-slate-100">
              <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-50 text-slate-600 rounded-lg text-xs font-bold hover:bg-slate-100 transition-all border border-slate-200 mb-2">
                <span className="material-symbols-outlined text-[18px]">download</span>
                Raw JSON Data
              </button>
              <button className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-50 text-slate-600 rounded-lg text-xs font-bold hover:bg-slate-100 transition-all border border-slate-200">
                <span className="material-symbols-outlined text-[18px]">share</span>
                Copy Share Link
              </button>
            </div>
          </div>
          <div className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/10">
            <p className="text-[10px] font-bold text-primary uppercase mb-1">Analysis Mode</p>
            <p className="text-xs text-slate-600">High-precision double float (64-bit) JLA backend.</p>
          </div>
        </aside>
      </div>
    </div>
  )
}

