import React, { useState } from 'react'
import MatrixAnalysisLayout from '../components/layout/MatrixAnalysisLayout'
import Breadcrumb from '../components/results/Breadcrumb'
import MatrixLatex from '../components/matrix/MatrixLatex'
import MatrixActionBar, { MatrixFooterBar } from '../components/matrix/MatrixActionBar'
import { useDiagnostics } from '../hooks/useDiagnostics'
import { usePrecisionUpdate } from '../hooks/usePrecisionUpdate'
import { formatComplex, formatNumber, formatDefiniteness } from '../utils/format'
import Latex from '../components/ui/Latex'
import { computeSpectralSeverity, computePerEigenvalueSeverity, SEVERITY_COLORS, computeNonOrthogonalEigenvectors, EIGEN_DISPLAY_MODES } from '../utils/spectralSeverity'
import { invertMatrix, getMatrixData } from '../utils/matrixOperations'

/**
 * Formats polynomial coefficients into a LaTeX-ready string.
 *
 * @param {Array<{real:number, imag:number}>} coeffs
 * @returns {string}
 */
function formatPolynomial(coeffs) {
  if (!Array.isArray(coeffs) || coeffs.length === 0) return '—'
  const degree = coeffs.length - 1
  const terms = []
  for (let i = 0; i < coeffs.length; i++) {
    const coeff = coeffs[i]
    const power = degree - i
    const coeffStr = typeof coeff === 'number' ? formatNumber(coeff, 3) : formatComplex(coeff, 3)
    if (power === 0) terms.push(`${coeffStr}`)
    else if (power === 1) terms.push(`${coeffStr} x`)
    else terms.push(`${coeffStr} x^{${power}}`)
  }
  return terms.join(' + ')
}

function parseComplexEntry(entry) {
  if (entry === null || entry === undefined) return [0, 0]
  if (typeof entry === 'number') return [entry, 0]
  if (Array.isArray(entry) && entry.length >= 2) return [Number(entry[0] ?? 0), Number(entry[1] ?? 0)]
  if (typeof entry === 'object') {
    const real = Number(entry.real ?? entry.r ?? entry.re ?? 0)
    const imag = Number(entry.imag ?? entry.i ?? entry.im ?? 0)
    return [real, imag]
  }
  if (typeof entry === 'string') {
    const s = entry.replace(/\s+/g, '')
    const match = s.match(/^([+-]?\d*\.?\d+)?([+-]?\d*\.?\d*)i?$/i)
    if (match) {
      const a = match[1] ? Number(match[1]) : 0
      const b = match[2] ? Number(match[2]) : 0
      return [a, b]
    }
    const num = Number(entry)
    if (!Number.isNaN(num)) return [num, 0]
  }
  return [0, 0]
}

export default function MatrixSpectralPage({ matrixString }) {
  // Subscribe to precision changes
  usePrecisionUpdate()
  
  const { diagnostics } = useDiagnostics(matrixString)
  const [mode, setMode] = useState(EIGEN_DISPLAY_MODES.UNIQUE_NO_REPS)
  
  const eigenvalues = diagnostics?.eigenvalues || []
  const eigenvectors = diagnostics?.eigenvectors
  const eigenspace = diagnostics?.eigenspace
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
  const hermitian = diagnostics?.hermitian
  const definiteness = diagnostics?.definiteness
  const spectralClass = diagnostics?.spectralClass || diagnostics?.spectral
  const spectrumLocation = diagnostics?.spectrumLocation
  const odeStability = diagnostics?.odeStability
  const discreteStability = diagnostics?.discreteStability
  const characteristic = formatPolynomial(diagnostics?.characteristicPolynomial)
  const isComplexDomain = diagnostics?.domain === 'C'

  const defectivity = diagnostics?.defectivity
  const diagOrDefective = (() => {
    if (diagonalizable === true) {
      return { label: 'Diagonalizable', ok: true, detail: 'Matrix admits a full eigenbasis.' }
    }
    if (diagonalizable === false || defectivity === true) {
      return { label: 'Defective', ok: false, detail: 'Insufficient linearly independent eigenvectors.' }
    }
    return { label: 'Unknown', ok: null, detail: 'Diagonalizability status unavailable.' }
  })()

  // Helper: normalize various vector/matrix shapes into a 2D numeric array suitable for MatrixDisplay
  const toMatrixDataFromVector = (vec) => {
    if (!vec) return null
    // Use the shared `parseComplexEntry` above

    // If wrapper objects provided
    if (vec.data && Array.isArray(vec.data)) {
      // if data is 2D array (rows), and looks like column vectors, return as-is
      if (Array.isArray(vec.data[0])) return vec.data
      // else if it's a flat array, convert to column vector (preserve complexness)
      const first = vec.data[0]
      const isComplex = first && (typeof first === 'object' || Array.isArray(first) || typeof first === 'string')
      if (isComplex) return vec.data.map(e => {
        const [r, i] = parseComplexEntry(e)
        return [{ real: r, imag: i }]
      })
      return vec.data.map(v => [v !== undefined && v !== null ? v : 0])
    }
    if (vec.vectors && Array.isArray(vec.vectors) && vec.vectors.length) {
      return toMatrixDataFromVector(vec.vectors[0])
    }
    // If vec is an array of entries
    if (Array.isArray(vec)) {
      if (vec.length === 0) return null
      const first = vec[0]
      const isComplex = first && (typeof first === 'object' || Array.isArray(first) || typeof first === 'string')
      if (isComplex) {
        return vec.map(v => {
          const [r, i] = parseComplexEntry(v)
          return [{ real: r, imag: i }]
        })
      }
      // nested arrays (already row arrays)
      if (Array.isArray(first)) return vec
      // simple numeric entries → column vector
      return vec.map(v => [ v !== undefined && v !== null ? v : 0 ])
    }
    // unsupported shape
    return null
  }

  // Helper: convert eigenspace vectors (array of column vectors) into row-major matrix for display
  const eigenspaceVectorsToMatrix = (vecs) => {
    if (!vecs || !Array.isArray(vecs) || vecs.length === 0) return null
    // vecs is array of vectors, each vector may be array of entries or object wrappers
    const firstVec = vecs[0]
    // normalize first vector to detect entry type
    let sampleEntry = null
    if (firstVec && Array.isArray(firstVec) && firstVec.length > 0) sampleEntry = firstVec[0]
    else if (firstVec && typeof firstVec === 'object') sampleEntry = firstVec

    // Use shared parseComplexEntry above

    const isComplex = (sampleEntry && typeof sampleEntry === 'object' && ('real' in sampleEntry || 'imag' in sampleEntry)) || (Array.isArray(sampleEntry) && sampleEntry.length === 2) || (typeof sampleEntry === 'string')

    // number of rows is length of a vector (after normalization)
    const numRows = Array.isArray(firstVec) ? firstVec.length : (firstVec && firstVec.data && Array.isArray(firstVec.data) ? firstVec.data.length : 0)
    if (!numRows) return null

    const result = []
    for (let i = 0; i < numRows; i++) {
      const row = []
      for (let j = 0; j < vecs.length; j++) {
        const v = vecs[j]
        let entry = null
        if (v && v.data && Array.isArray(v.data)) {
          entry = Array.isArray(v.data[0]) ? v.data[0][i] : v.data[i]
        } else if (Array.isArray(v)) {
          entry = v[i]
        } else if (v && v.vectors && Array.isArray(v.vectors) && v.vectors[0]) {
          entry = v.vectors[0][i]
        } else if (v && typeof v === 'object') {
          entry = v[i] !== undefined ? v[i] : null
        }

        if (isComplex) {
          const [a, b] = parseComplexEntry(entry)
          row.push({ real: a, imag: b })
        } else {
          row.push(entry !== undefined && entry !== null ? entry : 0)
        }
      }
      result.push(row)
    }
    return result
  }

  const matrixToDisplay = (matrix) => {
    if (!matrix) return null
    const data = matrix.data || matrix
    const imag = matrix.imag
    if (!Array.isArray(data)) return null
    // If explicit imag backing present, format each entry as a complex string
    if (imag) {
      return data.map((row, rIdx) => row.map((val, cIdx) => formatComplex({ real: val ?? 0, imag: imag?.[rIdx]?.[cIdx] ?? 0 }, 3)))
    }

    // If cells are already two-value arrays [real, imag], convert to complex strings
    const firstRow = data[0]
    if (firstRow && Array.isArray(firstRow) && Array.isArray(firstRow[0]) && firstRow[0].length === 2) {
      return data.map(row => row.map(cell => formatComplex({ real: cell[0] ?? 0, imag: cell[1] ?? 0 }, 3)))
    }

    // IMPORTANT: Only collapse flattened real/imag pairs when the domain is explicitly complex
    // AND the entries look like flattened complex pairs (each row has even length, double expected columns)
    // This prevents real even-column matrices from being misinterpreted as complex n/2-column matrices
    if (isComplexDomain && firstRow && Array.isArray(firstRow) && firstRow.length % 2 === 0 && firstRow.every(v => typeof v === 'number')) {
      // Additional check: compare against expected matrix dimensions if available
      const expectedCols = diagnostics?.columns
      if (expectedCols && firstRow.length === expectedCols * 2) {
        return data.map(row => {
          const out = []
          for (let c = 0; c < row.length; c += 2) {
            const a = row[c] ?? 0
            const b = row[c + 1] ?? 0
            out.push(formatComplex({ real: a, imag: b }, 3))
          }
          return out
        })
      }
    }

    return data
  }

  const toComplexMatrix = (matrix) => {
    if (!matrix) return null
    const data = matrix.data || matrix
    const imag = matrix.imag
    if (!Array.isArray(data) || !Array.isArray(data[0])) return null
    const rows = data.length
    const cols = data[0].length
    const realOut = Array.from({ length: rows }, () => new Array(cols).fill(0))
    const imagOut = Array.from({ length: rows }, () => new Array(cols).fill(0))

    if (imag) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          realOut[r][c] = data[r][c] ?? 0
          imagOut[r][c] = imag?.[r]?.[c] ?? 0
        }
      }
      return { data: realOut, imag: imagOut }
    }

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [re, im] = parseComplexEntry(data[r][c])
        realOut[r][c] = re
        imagOut[r][c] = im
      }
    }
    return { data: realOut, imag: imagOut }
  }

  const multiplyComplexMatrices = (a, b) => {
    if (!a || !b) return null
    const aRows = a.data.length
    const aCols = a.data[0]?.length || 0
    const bRows = b.data.length
    const bCols = b.data[0]?.length || 0
    if (!aRows || !aCols || !bRows || !bCols || aCols !== bRows) return null

    const outReal = Array.from({ length: aRows }, () => new Array(bCols).fill(0))
    const outImag = Array.from({ length: aRows }, () => new Array(bCols).fill(0))
    let anyImag = false

    for (let i = 0; i < aRows; i++) {
      for (let k = 0; k < aCols; k++) {
        const ar = a.data[i][k] ?? 0
        const ai = a.imag?.[i]?.[k] ?? 0
        if (ar === 0 && ai === 0) continue
        for (let j = 0; j < bCols; j++) {
          const br = b.data[k][j] ?? 0
          const bi = b.imag?.[k]?.[j] ?? 0
          const rr = ar * br - ai * bi
          const ii = ar * bi + ai * br
          outReal[i][j] += rr
          outImag[i][j] += ii
        }
      }
    }

    for (let r = 0; r < aRows; r++) {
      for (let c = 0; c < bCols; c++) {
        if (Math.abs(outImag[r][c]) > 1e-12) {
          anyImag = true
          break
        }
      }
      if (anyImag) break
    }

    return anyImag ? { data: outReal, imag: outImag } : { data: outReal }
  }

  // Collapse complex pairs (various shapes) into single a+bi strings for display
  const collapseComplexPairs = (data) => {
    if (!Array.isArray(data) || data.length === 0) return data
    const firstRow = data[0]
    if (!Array.isArray(firstRow) || firstRow.length === 0) return data
    // case: each cell is [real, imag]
    if (Array.isArray(firstRow[0]) && firstRow[0].length === 2) {
      return data.map(row => row.map(cell => formatComplex({ real: cell[0] ?? 0, imag: cell[1] ?? 0 }, 3)))
    }
    // case: flattened pairs per row - ONLY apply if domain is explicitly complex AND row length matches expected*2
    if (isComplexDomain && firstRow.every(v => typeof v === 'number') && firstRow.length % 2 === 0) {
      const expectedCols = diagnostics?.columns
      if (expectedCols && firstRow.length === expectedCols * 2) {
        return data.map(row => {
          const out = []
          for (let c = 0; c < row.length; c += 2) out.push(formatComplex({ real: row[c] ?? 0, imag: row[c + 1] ?? 0 }, 3))
          return out
        })
      }
    }
    // case: cell objects {real, imag}
    if (typeof firstRow[0] === 'object' && (firstRow[0].real !== undefined || firstRow[0].imag !== undefined)) {
      return data.map(row => row.map(cell => formatComplex({ real: cell.real ?? 0, imag: cell.imag ?? 0 }, 3)))
    }
    return data
  }

  const formatLatexEntry = (entry) => {
    if (entry === null || entry === undefined) return '\\text{—}'
    if (typeof entry === 'string') return entry
    if (Array.isArray(entry) && entry.length === 2) {
      return formatComplex({ real: Number(entry[0] ?? 0), imag: Number(entry[1] ?? 0) }, 3)
    }
    if (typeof entry === 'object' && (entry.real !== undefined || entry.imag !== undefined || entry.r !== undefined || entry.i !== undefined)) {
      return formatComplex({ real: Number(entry.real ?? entry.r ?? 0), imag: Number(entry.imag ?? entry.i ?? 0) }, 3)
    }
    if (typeof entry === 'number') return formatNumber(entry, 3)
    const [r, i] = parseComplexEntry(entry)
    return formatComplex({ real: r, imag: i }, 3)
  }

  const toBasisLatex = (matrix) => {
    if (!Array.isArray(matrix) || matrix.length === 0) return ''
    const cols = matrix[0]?.length || 0
    if (!cols) return ''
    const parts = Array.from({ length: cols }, (_, cIdx) => {
      const entries = matrix.map(row => formatLatexEntry(row?.[cIdx]))
      return `\\begin{pmatrix}${entries.join('\\\\')}\\end{pmatrix}`
    })
    return `\\left\\{ ${parts.join(',')} \\right\\}`
  }

  const extractEigenvectorColumn = (matrix, colIndex) => {
    if (!matrix) return null
    const data = matrix.data || matrix
    const imag = matrix.imag
    if (!Array.isArray(data) || data.length === 0) return null
    return data.map((row, rIdx) => {
      const entry = row && row[colIndex]
      // handle explicit imag backing
      if (imag) {
        const real = row && row[colIndex] !== undefined ? row[colIndex] : 0
        const im = imag?.[rIdx]?.[colIndex] ?? 0
        return { real, imag: im }
      }
      const [r, im] = parseComplexEntry(entry)
      return { real: r, imag: im }
    })
  }

  const eigenvectorsDisplay = matrixToDisplay(eigenvectors)
  // For eigenbasis display, prefer eigenspace but fall back to eigenvectors if diagonalizable
  const eigenbasisDisplay = (() => {
    const espaceDisplay = matrixToDisplay(eigenspace)
    if (espaceDisplay && espaceDisplay.length > 0) return espaceDisplay
    // If diagonalizable is true but no eigenspace data, use eigenvectors as the eigenbasis
    if (diagonalizable && eigenvectorsDisplay && eigenvectorsDisplay.length > 0) {
      return eigenvectorsDisplay
    }
    return null
  })()
  
  // Compute which eigenvector columns are non-orthogonal to all others
  const nonOrthogonalIndices = computeNonOrthogonalEigenvectors(diagnostics)
  
  // Attempt to infer a diagonalization if backend omitted it but eigenvectors/eigenvalues form a full basis
  let inferredDiagP = null
  let inferredDiagD = null
  let inferredDiagPInverse = null
  try {
    if ((!diagP || !diagD || !diagPInverse) && eigenvectors) {
      const numericP = getMatrixData(eigenvectors) || (Array.isArray(eigenvectors) ? eigenvectors : null)
      if (numericP && numericP.length > 0) {
        const n = numericP.length
        const cols = numericP[0]?.length || 0
        if (cols === n) {
          const pinv = invertMatrix(numericP)
          if (pinv) {
            inferredDiagP = eigenvectors
            // build diagonal D from eigenvalues (keep complex parts if present)
            const dMat = Array.from({ length: n }, () => new Array(n).fill(0))
            for (let i = 0; i < n; i++) {
              const ev = eigenvalues[i] || eigenvalues[0] || { real: 0, imag: 0 }
              const real = (ev && (ev.real !== undefined)) ? ev.real : (typeof ev === 'number' ? ev : 0)
              const imag = (ev && (ev.imag !== undefined)) ? ev.imag : 0
              dMat[i][i] = (imag === 0) ? real : { real, imag }
            }
            inferredDiagD = dMat
            inferredDiagPInverse = pinv
          }
        }
      }
    }
  } catch (e) {
    // fall back silently if inference fails
  }
  
  const diagProduct = (() => {
    // allow inferred diagonalization if backend didn't provide one but we can construct it
    const pRaw = diagP || inferredDiagP
    const dRaw = diagD || inferredDiagD
    const pinvRaw = diagPInverse || inferredDiagPInverse
    if (!pRaw || !dRaw || !pinvRaw) return null
    const p = toComplexMatrix(pRaw)
    const d = toComplexMatrix(dRaw)
    const pinv = toComplexMatrix(pinvRaw)
    if (!p || !d || !pinv) return null
    const pd = multiplyComplexMatrices(p, d)
    return multiplyComplexMatrices(pd, pinv)
  })()
  const diagProductDisplay = matrixToDisplay(diagProduct)

  // Compute spectral severity
  const overallSeverity = computeSpectralSeverity(diagnostics)

  // Process eigenvalues for display
  const tol = 1e-9
  const alg = diagnostics?.algebraicMultiplicity || []
  const geo = diagnostics?.geometricMultiplicity || []
  
  // Use backend eigenInformationPerValue if available, otherwise construct from eigenvalues
  let perEigenInfo = diagnostics?.eigenInformationPerValue || []
  if (perEigenInfo.length === 0 && eigenvalues.length > 0) {
    // Construct minimal perEigenInfo from eigenvalues
    const seenEigenvalues = new Map()
    eigenvalues.forEach((ev, idx) => {
      const key = `${ev?.real ?? ev}_${ev?.imag ?? 0}`
      if (!seenEigenvalues.has(key)) {
        seenEigenvalues.set(key, {
          eigenvalue: ev,
          algebraicMultiplicity: alg[idx] ?? null,
          geometricMultiplicity: geo[idx] ?? null,
          dimension: geo[idx] ?? null,
          representativeEigenvector: null,
          eigenspace: null,
          eigenbasis: null
        })
      }
    })
    perEigenInfo = Array.from(seenEigenvalues.values())
  }
  
  // Compute unique eigenvalues based on display mode
  const uniqueEigenvalues = (() => {
    if (mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS) {
      return eigenvalues
    }
    // UNIQUE_NO_REPS mode: filter duplicates
    const unique = []
    for (const lambda of eigenvalues) {
      const exists = unique.some(u => 
        Math.hypot((u?.real ?? 0) - (lambda?.real ?? 0), (u?.imag ?? 0) - (lambda?.imag ?? 0)) <= tol
      )
      if (!exists) unique.push(lambda)
    }
    return unique
  })()

  const findOriginalIndex = (lambda) => {
    for (let i = 0; i < eigenvalues.length; i++) {
      if (Math.hypot((eigenvalues[i]?.real ?? 0) - (lambda?.real ?? 0), 
                     (eigenvalues[i]?.imag ?? 0) - (lambda?.imag ?? 0)) <= tol) {
        return i
      }
    }
    return -1
  }

  // Track critical artifacts for highlighting
  const criticalArtifacts = {
    eigenvectorMatrix: overallSeverity?.level === 'critical' || nonOrthogonalIndices?.size > 0,
    eigendecomposition: diagOrDefective?.ok === false || overallSeverity?.level === 'critical'
  }

  return (
    <MatrixAnalysisLayout
      matrixString={matrixString}
      diagnostics={diagnostics}
      activeTab="spectral"
      title="Spectral Analysis"
      subtitle="Eigendecomposition and spectral characteristic identification"
      breadcrumbs={<Breadcrumb items={[{ label: 'Dashboard', href: '#' }, { label: 'Spectral Analysis' }]} />}
      actions={<MatrixActionBar matrixString={matrixString} diagnostics={diagnostics} />}
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
                  <div className="flex items-center gap-2 ml-3">
                    <label className="text-[11px] text-slate-500">Show representatives</label>
                    <select value={mode} onChange={e => setMode(e.target.value)} className="text-sm border rounded px-2 py-1">
                      <option value={EIGEN_DISPLAY_MODES.UNIQUE_NO_REPS}>Unique (no reps)</option>
                      <option value={EIGEN_DISPLAY_MODES.ALL_WITH_REPS}>All (with reps)</option>
                    </select>
                  </div>
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
                  const evSeverity = computePerEigenvalueSeverity(lambda, origIdx >= 0 ? origIdx : idx, diagnostics, { displayMode: mode })
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
                  
                  // Compute algVal and geoVal for this eigenvalue
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
                  
                  // Prefer explicit eigenbasis (list of basis vectors). Fall back to eigenspace if no eigenbasis provided.
                  const eigenbasisVectors = info?.eigenbasis?.vectors || null
                  const eigenspaceVectors = info?.eigenspace?.vectors || null
                  const eigenbasisDim = info?.dimension ?? info?.geometricMultiplicity ?? (eigenbasisVectors ? eigenbasisVectors.length : (eigenspaceVectors ? eigenspaceVectors.length : 0))
                  // Convert eigenspace vectors into a row-major matrix for display when needed
                  const eigenbasisAsColumns = eigenspaceVectorsToMatrix(eigenspaceVectors)
                  // In ALL_WITH_REPS mode, idx maps directly to eigenvalues[idx]; in UNIQUE_NO_REPS mode, find first occurrence
                  const origIdx = mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS ? idx : (lambda ? findOriginalIndex(lambda) : -1)
                  const evSeverity = lambda ? computePerEigenvalueSeverity(lambda, origIdx >= 0 ? origIdx : idx, diagnostics, { displayMode: mode }) : { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }
                  const algMultiplicity = algVal
                  const geoMultiplicity = geoVal
                  const multiplicityMismatch = algMultiplicity !== null && geoMultiplicity !== null && algMultiplicity !== geoMultiplicity
                  const multiplicityColor = multiplicityMismatch ? SEVERITY_COLORS.severe : SEVERITY_COLORS.safe
                  
                  // Determine the representative eigenvector for this eigenvalue occurrence.
                  // In ALL_WITH_REPS mode with repeated eigenvalues, each occurrence should use the
                  // corresponding column from the eigenvector matrix (at index `idx`), not the same
                  // representativeEigenvector from perEigenInfo (which is shared for all occurrences).
                  let repMatrix = null
                  if (mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS) {
                    // Always extract from eigenvector matrix by column index in ALL_WITH_REPS mode
                    try {
                      if (idx >= 0) {
                        const col = extractEigenvectorColumn(eigenvectors, idx)
                        repMatrix = toMatrixDataFromVector(col)
                      }
                    } catch (e) {
                      // ignore extraction errors
                    }
                  }
                  // Fallback: use info's representativeEigenvector or extract from eigenvector matrix
                  if (!repMatrix) {
                    const rep = info?.representativeEigenvector || null
                    repMatrix = toMatrixDataFromVector(rep)
                    if (!repMatrix && rep === null) {
                      try {
                        const eigenvectorIndex = lambda ? findOriginalIndex(lambda) : -1
                        if (eigenvectorIndex >= 0) {
                          const col = extractEigenvectorColumn(eigenvectors, eigenvectorIndex)
                          repMatrix = toMatrixDataFromVector(col)
                        }
                      } catch (e) {
                        // ignore fallback errors
                      }
                    }
                  }
                  return (
                    <div key={`basis-${idx}`} className="mt-2 p-3 border-l-4 rounded-r" style={{ borderColor: evSeverity.color, backgroundColor: `${evSeverity.color}08` }}>
                      <div className="flex items-center justify-between mb-2"><div className="text-xs text-slate-500">Eigenspace for {lambda ? <Latex tex={`\\lambda = ${formatComplex(lambda,3)}`} /> : <span>λ</span>}</div><span className="text-[9px] px-1.5 py-0.5 rounded font-bold text-white" style={{ backgroundColor: evSeverity.color }} title={evSeverity.issues?.join('; ')}>{evSeverity.label}</span></div>
                      <div className="flex gap-4 items-start">
                        <div className="flex-1">
                          {eigenbasisVectors ? (
                            <>
                              <div className="text-[10px] text-slate-400 mb-1">Eigenbasis (dim: {eigenbasisDim})</div>
                              <div style={multiplicityMismatch ? { color: evSeverity.color } : {}}>
                                {(() => {
                                  // Build LaTeX for set of column vectors \left\{ \begin{pmatrix} ... \\ ... \end{pmatrix}, ... \right\}
                                  const rows = eigenbasisVectors[0] && eigenbasisVectors[0].length ? eigenbasisVectors[0].length : (Array.isArray(eigenbasisVectors[0]) ? eigenbasisVectors[0].length : 0)
                                  const vecs = eigenbasisVectors
                                  const parts = vecs.map(v => {
                                    const entries = (Array.isArray(v) ? v : (v.data && Array.isArray(v.data) ? v.data : [])).map(e => {
                                      const [re, im] = parseComplexEntry(e)
                                      return formatComplex({ real: re, imag: im }, 3)
                                    })
                                    return `\\begin{pmatrix}${entries.join('\\\\')}\\end{pmatrix}`
                                  })
                                  const latex = `\\left\\{ ${parts.join(',')} \\right\\}`
                                  return <Latex tex={latex} />
                                })()}
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="text-[10px] text-slate-400 mb-1">Eigenspace (dim: {eigenbasisDim})</div>
                              {eigenbasisAsColumns ? (
                                <div style={multiplicityMismatch ? { color: evSeverity.color } : {}}>
                                  <MatrixLatex data={collapseComplexPairs(eigenbasisAsColumns)} className="text-sm math-font" precision={3} />
                                </div>
                              ) : (
                                <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>—</div>
                              )}
                            </>
                          )}
                        </div>
                        {mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS && diagOrDefective.ok !== false && (
                          <div className="flex-1">
                            <div className="text-[10px] text-slate-400 mb-1">Representative Eigenvector</div>
                            {repMatrix ? (
                              <div style={multiplicityMismatch ? { color: evSeverity.color } : {}}>
                                {(() => {
                                  // repMatrix is row-major array of single-column rows: [[cell],[cell],...]
                                  const entries = repMatrix.map(row => {
                                    const cell = Array.isArray(row) ? row[0] : row
                                    if (cell == null) return formatComplex({ real: 0, imag: 0 }, 3)
                                    if (typeof cell === 'object' && (cell.real !== undefined || cell.imag !== undefined)) {
                                      return formatComplex({ real: Number(cell.real ?? 0), imag: Number(cell.imag ?? 0) }, 3)
                                    }
                                    if (Array.isArray(cell) && cell.length === 2) {
                                      return formatComplex({ real: Number(cell[0] ?? 0), imag: Number(cell[1] ?? 0) }, 3)
                                    }
                                    const [r, i] = parseComplexEntry(cell)
                                    return formatComplex({ real: r, imag: i }, 3)
                                  })
                                  const latex = `\\begin{pmatrix}${entries.join('\\\\')}\\end{pmatrix}`
                                  return <Latex tex={latex} />
                                })()}
                              </div>
                            ) : (
                              <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>—</div>
                            )}
                          </div>
                        )}
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
                  // Sum algebraic multiplicities from perEigenInfo (unique eigenvalues) if available
                  // This correctly handles scalar/identity matrices where all eigenvalues are the same
                  const sumAlg = perEigenInfo && perEigenInfo.length > 0 
                    ? perEigenInfo.reduce((sum, info) => sum + (info?.algebraicMultiplicity ?? 0), 0)
                    : alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
                  const matrixDim = diagnostics?.rows ?? diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
                  // Only show critical if we have valid data and there's a true mismatch
                  const showCritical = sumAlg > 0 && matrixDim > 0 && sumAlg !== matrixDim
                  return showCritical ? (
                    <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
                  ) : null
                })()}
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center">
              {diagOrDefective.ok === false ? (
                <div className="text-xs text-center" style={{ color: SEVERITY_COLORS.critical }}>
                  Eigenvector matrix not displayed for defective matrices.
                </div>
              ) : (
              <div className="py-4 px-4" style={criticalArtifacts.eigenvectorMatrix ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 8 } : {}}>
                <MatrixLatex data={eigenvectorsDisplay} className="math-font text-sm font-medium" precision={3} />
              </div>
              )}
            </div>
            <div className="mt-6 space-y-4"><div className="h-[1px] bg-slate-200"></div><div><h4 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-tighter">Quick Summary</h4><ul className="space-y-4"><li className="flex items-start gap-3"><span className="material-symbols-outlined text-primary text-sm mt-1">check_circle</span><div><p className="text-sm font-bold"><Latex tex={'\\text{Spectral Radius} \\ (\\rho)'} /></p><p className="text-lg font-bold math-font text-primary">{formatNumber(diagnostics?.spectralRadius, 4)}</p><p className="text-xs text-slate-500">Maximum absolute eigenvalue.</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${diagOrDefective.ok === true ? 'text-emerald-600' : diagOrDefective.ok === false ? 'text-rose-500' : 'text-amber-500'}`}>{diagOrDefective.ok === true ? 'check_circle' : diagOrDefective.ok === false ? 'error' : 'help'}</span><div><p className="text-sm font-bold">{diagOrDefective.label}</p><p className="text-xs text-slate-500">{diagOrDefective.detail}</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${normal ? 'text-emerald-600' : 'text-amber-500'}`}>{normal ? 'check_circle' : 'warning'}</span><div><p className="text-sm font-bold">{normal ? 'Normal' : 'Non-normal'}</p><p className="text-xs text-slate-500">{normal ? 'Commutes with its adjoint.' : 'Does not commute with its adjoint.'}</p></div></li><li className="flex items-start gap-3"><span className={`material-symbols-outlined text-sm mt-1 ${hermitian ? 'text-emerald-600' : 'text-amber-500'}`}>{hermitian ? 'check_circle' : 'warning'}</span><div><p className="text-sm font-bold">{hermitian ? 'Hermitian' : 'Non-Hermitian'}</p><p className="text-xs text-slate-500">{hermitian ? 'Equals its adjoint.' : 'Does not equal its adjoint.'}</p></div></li><li className="flex items-start gap-3"><div><p className="text-sm font-bold">Orthonormal: {orthonormal ? 'Yes' : 'No'}</p><p className="text-sm font-bold">Orthogonal: {orthogonal ? 'Yes' : 'No'}</p><p className="text-sm font-bold">Unitary: {unitary ? 'Yes' : 'No'}</p></div></li><li className="flex items-start gap-3"><div><p className="text-sm font-bold">Spectrum Location: <span style={spectrumLocation === undefined || spectrumLocation === null ? { color: SEVERITY_COLORS.critical } : {}}>{spectrumLocation ?? 'Unknown'}</span></p><p className="text-sm font-bold">ODE Stability: <span style={odeStability === undefined || odeStability === null ? { color: SEVERITY_COLORS.critical } : {}}>{odeStability ?? 'Unknown'}</span></p><p className="text-sm font-bold">Discrete Stability: <span style={discreteStability === undefined || discreteStability === null ? { color: SEVERITY_COLORS.critical } : {}}>{discreteStability ?? 'Unknown'}</span></p></div></li><li className="flex items-start gap-3"><div><p className="text-sm font-bold">Definiteness: <span style={definiteness === undefined || definiteness === null ? { color: SEVERITY_COLORS.critical } : {}}>{formatDefiniteness(definiteness)}</span></p><p className="text-sm font-bold">Spectral Class: <span style={spectralClass === undefined || spectralClass === null ? { color: SEVERITY_COLORS.critical } : {}}>{spectralClass ?? 'Unknown'}</span></p></div></li></ul></div></div>
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
                // Sum algebraic multiplicities from perEigenInfo (unique eigenvalues) if available
                const sumAlg = perEigenInfo && perEigenInfo.length > 0 
                  ? perEigenInfo.reduce((sum, info) => sum + (info?.algebraicMultiplicity ?? 0), 0)
                  : alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
                const matrixDim = diagnostics?.rows ?? diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
                const showCritical = sumAlg > 0 && matrixDim > 0 && sumAlg !== matrixDim
                return showCritical ? (
                  <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
                ) : null
              })()}
            </div>
            <div className="text-xs" style={{ color: diagStatus ? undefined : (diagonalizable ? '#388E3C' : SEVERITY_COLORS.critical) }}>
              {diagStatus ?? (diagonalizable ? 'Diagonalizable (data loading...)' : 'Unavailable')}
            </div>
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 items-start">
            {diagOrDefective.ok === false ? (
              <div className="col-span-full text-xs text-center" style={{ color: SEVERITY_COLORS.critical }}>
                Eigendecomposition not available for defective matrices.
              </div>
            ) : (
              <>
            <div>
              <div className="text-[10px] text-slate-400 mb-2">P (eigenvector matrix)</div>
              {(diagP || inferredDiagP) ? (
                <div style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}}>
                  <MatrixLatex data={matrixToDisplay(diagP || inferredDiagP)} className="math-font text-sm" precision={3} />
                </div>
              ) : (
                <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Eigenvector matrix unavailable.</div>
              )}
            </div>
            <div>
              <div className="text-[10px] text-slate-400 mb-2">D (diagonal eigenvalue matrix)</div>
              {(diagD || inferredDiagD) ? (
                <div style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}}>
                  <MatrixLatex data={matrixToDisplay(diagD || inferredDiagD)} className="math-font text-sm" precision={3} />
                </div>
              ) : (
                <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Diagonal matrix unavailable.</div>
              )}
            </div>
            <div>
              <div className="text-[10px] text-slate-400 mb-2"><Latex tex={'P^{-1}'} /> (inverse eigenvector matrix)</div>
              {(diagPInverse || inferredDiagPInverse) ? (
                <div style={criticalArtifacts.eigendecomposition ? { boxShadow: '0 0 0 2px rgba(220,38,38,0.08)', borderRadius: 6 } : {}}>
                  <MatrixLatex data={matrixToDisplay(diagPInverse || inferredDiagPInverse)} className="math-font text-sm" precision={3} />
                </div>
              ) : (
                <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Inverse matrix unavailable.</div>
              )}
            </div>
            <div>
              <div className="text-[10px] text-slate-400 mb-2">Reconstruction <Latex tex={'PDP^{-1}'} /></div>
              {diagProductDisplay ? (
                <MatrixLatex data={diagProductDisplay} className="math-font text-sm" precision={2} />
              ) : (
                <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>Reconstruction unavailable.</div>
              )}
            </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Global Eigenbasis card */}
      <div className="p-8">
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex items-center gap-2">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Eigenbasis</h4>
            {(() => {
              // Sum algebraic multiplicities from perEigenInfo (unique eigenvalues) if available
              const sumAlg = perEigenInfo && perEigenInfo.length > 0 
                ? perEigenInfo.reduce((sum, info) => sum + (info?.algebraicMultiplicity ?? 0), 0)
                : alg?.reduce((sum, val) => sum + (val ?? 0), 0) ?? 0
              const matrixDim = diagnostics?.rows ?? diagnostics?.cols ?? diagnostics?.n ?? eigenvalues?.length ?? 0
              const showCritical = sumAlg > 0 && matrixDim > 0 && sumAlg !== matrixDim
              return showCritical ? (
                <span className="text-[10px] px-2 py-0.5 rounded font-bold text-white" style={{ backgroundColor: SEVERITY_COLORS.critical }} title="Sum of algebraic multiplicities does not equal matrix dimension">CRITICAL</span>
              ) : null
            })()}
          </div>
          <div className="p-6">
            {diagOrDefective.ok === false ? (
              <div className="text-xs" style={{ color: SEVERITY_COLORS.critical }}>
                Eigenbasis does not exist (matrix is defective).
              </div>
            ) : (!eigenbasisDisplay || eigenbasisDisplay.length === 0) ? (
              <div className="text-xs" style={{ color: diagonalizable ? '#388E3C' : SEVERITY_COLORS.critical }}>
                {diagonalizable 
                  ? 'Diagonalizable - eigenbasis exists (any orthonormal set of eigenvectors forms an eigenbasis)' 
                  : 'Eigenbasis unavailable (matrix is defective).'}
              </div>
            ) : (
              <Latex tex={toBasisLatex(eigenbasisDisplay)} className="text-sm math-font" />
            )}
          </div>
        </div>
      </div>
      <MatrixFooterBar matrixString={matrixString} diagnostics={diagnostics} />
    </MatrixAnalysisLayout>
  )
}



