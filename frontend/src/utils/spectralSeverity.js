/**
 * Spectral severity classification for eigenvalue/eigenvector analysis.
 * Returns severity level and color based on various numerical stability indicators.
 */

export const SEVERITY_COLORS = {
  critical: '#D32F2F',    // critical/dangerous
  severe: '#F57C00',      // severe/unstable
  moderate: '#FBC02D',    // moderate/caution
  mild: '#1976D2',        // mild/informational
  safe: '#388E3C'         // healthy/ideal
}

export const SEVERITY_LABELS = {
  critical: 'Critical 🚨',
  severe: 'Severe ✴️',
  moderate: 'Moderate ⚠️',
  mild: 'Mild ℹ️',
  safe: 'Safe ✅'
}

// Display mode toggle for how eigen-information is presented.
export const EIGEN_DISPLAY_MODES = {
  ALL_WITH_REPS: 'all_with_reps',       // show every eigenvalue and representative eigenvector
  UNIQUE_NO_REPS: 'unique_no_reps'      // show unique eigenvalues and ignore representative eigenvectors
}

// Default mode (can be changed at runtime via `setEigenDisplayMode`).
// Default to hiding representative eigenvectors in the UI
export let eigenDisplayMode = EIGEN_DISPLAY_MODES.UNIQUE_NO_REPS
export function setEigenDisplayMode(mode) {
  if (mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS || mode === EIGEN_DISPLAY_MODES.UNIQUE_NO_REPS) {
    eigenDisplayMode = mode
  }
}

/**
 * Compute overall spectral severity based on diagnostics.
 *
 * @param {any} diagnostics
 * @returns {{level: string, color: string, label: string, issues: string[], criticalArtifacts?: any}}
 */
export function computeSpectralSeverity(diagnostics, options = {}) {
  if (!diagnostics) return { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }
  const mode = options.displayMode || eigenDisplayMode

  const issues = []
  let severity = 'safe'

  const eigenvalues = diagnostics.eigenvalues || []
  const eigenvectors = normalizeMatrix(diagnostics.eigenvectors)
  const alg = diagnostics.algebraicMultiplicity || []
  const geom = diagnostics.geometricMultiplicity || []
  const perEigen = diagnostics.eigenInformationPerValue || null
  let criticalArtifacts = { eigenbasis: false, eigenvectorMatrix: false, eigendecomposition: false }
  const diagonalizable = diagnostics.diagonalizable
  const normal = diagnostics.normal
  const orthogonal = diagnostics.orthogonal
  const conditionNumber = diagnostics.conditionNumber

  // CRITICAL checks
  // 1. No eigenbasis (matrix is defective)
  if (diagonalizable === false) {
    // Compute sum of geometric multiplicities per unique eigenvalue
    const tol = 1e-9
    const counted = new Array(eigenvalues.length).fill(false)
    let totalGeom = 0
    for (let i = 0; i < eigenvalues.length; i++) {
      if (counted[i]) continue
      const lambda = eigenvalues[i]
      for (let j = i; j < eigenvalues.length; j++) {
        if (!counted[j]) {
          const other = eigenvalues[j]
          const dist = Math.hypot((lambda?.real ?? lambda ?? 0) - (other?.real ?? other ?? 0), (lambda?.imag ?? 0) - (other?.imag ?? 0))
          if (dist <= tol) {
            counted[j] = true
          }
        }
      }
      totalGeom += (geom[i] ?? 0)
    }
    const n = diagnostics.rows || diagnostics.n || eigenvalues.length
    if (totalGeom < n) {
      severity = 'critical'
      issues.push(`Eigenbasis does not exist (${totalGeom} independent eigenvectors < ${n} dimension)`)
    }
  }

  // 1b. Representative eigenvectors for unique eigenvalues are not independent
  try {
    // Only use representative eigenvectors when display mode requests them.
    if (mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS && perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
      const reps = perEigen.map(p => p.representativeEigenvector).filter(Boolean)
      if (reps.length > 0) {
        const indep = estimateIndependentColumns(reps)
        const n = diagnostics.rows || eigenvalues.length || 0
        if (indep < n) {
          severity = 'critical'
          criticalArtifacts = { eigenbasis: true, eigenvectorMatrix: true, eigendecomposition: true }
          issues.push(`Representative eigenvectors span ${indep} < ${n}; eigendecomposition unstable or incomplete`)
        } else if (indep < reps.length) {
          if (severity === 'safe' || severity === 'mild' || severity === 'moderate') severity = 'severe'
          issues.push(`Representative eigenvectors are linearly dependent (rank ${indep} < ${reps.length})`)
        }
      }
    }
  } catch (e) {
    // ignore numeric failures
  }

  // 2. Geometric multiplicity much smaller than algebraic (strongly defective)
  // Note: geom[i] can be -1 if computation failed, treat as unknown rather than defective
  for (let i = 0; i < alg.length; i++) {
    if (alg[i] && alg[i] > 1 && geom[i] !== undefined && geom[i] !== null && geom[i] >= 0 && geom[i] < alg[i] * 0.5) {
      severity = severity === 'safe' ? 'critical' : severity
      issues.push(`Eigenvalue ${i + 1}: geometric multiplicity (${geom[i]}) << algebraic (${alg[i]})`)
    }
  }

  // 3. V matrix nearly singular (if we have eigenvectors)
  if (eigenvectors && eigenvectors.length > 0) {
    const det = computeDeterminant(eigenvectors)
    if (Math.abs(det) < 1e-10) {
      severity = severity === 'safe' ? 'critical' : severity
      issues.push(`Eigenvector matrix is nearly singular (det ≈ ${det.toExponential(2)})`)
    }
    
    // High condition number of V
    const vCond = estimateConditionNumber(eigenvectors)
    if (vCond > 1e12) {
      severity = severity === 'safe' || severity === 'mild' ? 'critical' : severity
      issues.push(`Eigenvector matrix is ill-conditioned (κ ≈ ${vCond.toExponential(2)})`)
    }
  }

  // SEVERE checks
  // 1. Eigenvector sensitivity due to close eigenvalues (only for non-normal, non-diagonalizable matrices)
  // For normal matrices or diagonalizable matrices with repeated eigenvalues, eigenvectors are stable.
  const gaps = computeEigenvalueGaps(eigenvalues)
  for (let i = 0; i < gaps.length; i++) {
    if (gaps[i] < 1e-6 && !normal && diagonalizable === false) {
      if (severity === 'safe' || severity === 'mild' || severity === 'moderate') {
        severity = 'severe'
      }
      issues.push(`Eigenvalue ${i + 1}: small gap (${gaps[i].toExponential(2)}), eigenvectors may be volatile`)
    }
  }

  // 2. Repeated eigenvalues with deficient geometric multiplicity (truly defective)
  // Only flag if geometric < algebraic (defective), not just for repeated eigenvalues
  for (let i = 0; i < alg.length; i++) {
    if (alg[i] > 1 && geom[i] !== undefined && geom[i] !== null && geom[i] >= 0 && geom[i] < alg[i]) {
      if (severity === 'safe' || severity === 'mild' || severity === 'moderate') {
        severity = 'severe'
      }
      issues.push(`Eigenvalue ${i + 1}: defective (alg=${alg[i]}, geo=${geom[i]})`)
    }
  }

  // MODERATE checks
  // 1. Tightly clustered eigenvalues
  const clusters = findEigenvalueClusters(eigenvalues)
  if (clusters.length > 0) {
    if (severity === 'safe' || severity === 'mild') {
      severity = 'moderate'
    }
    issues.push(`${clusters.length} eigenvalue cluster(s) detected`)
  }

  // 2. Large eigenvalue spread
  const spread = computeEigenvalueSpread(eigenvalues)
  if (spread > 1e6) {
    if (severity === 'safe' || severity === 'mild') {
      severity = 'moderate'
    }
    issues.push(`Large eigenvalue spread (${spread.toExponential(2)})`)
  }

  // 3. Eigenvalues near zero
  const nearZeroCount = eigenvalues.filter(ev => Math.abs(ev.real) < 1e-8 && Math.abs(ev.imag) < 1e-8).length
  if (nearZeroCount > 0) {
    if (severity === 'safe' || severity === 'mild') {
      severity = 'moderate'
    }
    issues.push(`${nearZeroCount} eigenvalue(s) near zero`)
  }

  // MILD checks
  // 1. Non-orthogonal eigenvectors - eigenvectors that are not orthogonal to each other
  // indicate potential numerical sensitivity even if the matrix is otherwise healthy
  if (eigenvectors && eigenvectors.length > 0) {
    const orthoScore = checkOrthogonality(eigenvectors)
    if (orthoScore < 0.9) {
      // Non-orthogonal eigenvectors should always be flagged as mild at minimum
      if (severity === 'safe') {
        severity = 'mild'
      }
      issues.push(`Eigenvectors are not orthogonal (score: ${orthoScore.toFixed(3)})`)
    }
  }

  // SAFE conditions (override if all checks pass)
  if (issues.length === 0) {
    const allMatch = alg.every((a, i) => a === geom[i])
    if (allMatch && (orthogonal || normal)) {
      severity = 'safe'
      issues.push('All geometric = algebraic multiplicities, eigenvectors well-separated')
    }
  }

  return {
    level: severity,
    color: SEVERITY_COLORS[severity],
    label: SEVERITY_LABELS[severity],
    issues: issues,
    criticalArtifacts: criticalArtifacts
  }
}

/**
 * Compute per-eigenvalue severity.
 *
 * @param {{real:number, imag:number}} eigenvalue
 * @param {number} index
 * @param {any} diagnostics
 * @returns {{level: string, color: string, label: string, issues: string[]}}
 */
export function computePerEigenvalueSeverity(eigenvalue, index, diagnostics, options = {}) {
  if (!diagnostics) return { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }
  const mode = options.displayMode || eigenDisplayMode

  const issues = []
  let severity = 'safe'

  const alg = diagnostics.algebraicMultiplicity || []
  const geom = diagnostics.geometricMultiplicity || []
  const perEigen = diagnostics.eigenInformationPerValue || null
  const eigenvalues = diagnostics.eigenvalues || []
  // matrix dimension (rows) if available
  const n = diagnostics.rows || eigenvalues.length || 0

  // If per-eigen array exists, try to find the matching per-entry for this eigenvalue/index
  let perEntry = null
  try {
    if (perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
      // prefer matching by exact index if diagnostics populated that way
      if (index >= 0 && index < (eigenvalues.length || 0)) {
        const target = eigenvalues[index]
        for (const p of perEigen) {
          if (!p || !p.eigenvalue) continue
          const v = p.eigenvalue
          if (Math.hypot((v.real ?? 0) - (target.real ?? 0), (v.imag ?? 0) - (target.imag ?? 0)) <= 1e-8) { perEntry = p; break }
        }
      }
      // fallback: match by proximity to the provided eigenvalue argument
      if (!perEntry && eigenvalue) {
        for (const p of perEigen) {
          if (!p || !p.eigenvalue) continue
          const v = p.eigenvalue
          if (Math.hypot((v.real ?? 0) - (eigenvalue.real ?? 0), (v.imag ?? 0) - (eigenvalue.imag ?? 0)) <= 1e-8) { perEntry = p; break }
        }
      }
    }
  } catch (e) {
    // ignore
  }

  // gather representative vectors for dependency/orthogonality checks
  let reps = null
  let myRep = null
  try {
    const perEigen = diagnostics.eigenInformationPerValue || null
    if (mode === EIGEN_DISPLAY_MODES.ALL_WITH_REPS && perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
      reps = perEigen.map(p => p.representativeEigenvector).filter(Boolean)
      myRep = perEntry?.representativeEigenvector || null
    }
  } catch (e) { reps = null; myRep = null }

  // 1) SEVERE: Representative eigenvector is linearly dependent on others
  if (myRep && reps && reps.length > 1) {
    try {
      const others = reps.filter(r => r !== myRep)
      const indepBefore = estimateIndependentColumns(others)
      const indepWithThis = estimateIndependentColumns(reps)
      if (indepWithThis === indepBefore) {
        // This representative does not add to the rank - it's dependent
        severity = 'severe'
        issues.push('Representative eigenvector is linearly dependent on others')
        return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
      }
    } catch (e) {
      // ignore numeric failures
    }
  }

  // 2) SEVERE: Eigenvector sensitive to perturbations (small eigenvalue gap, non-normal AND non-diagonalizable)
  // For diagonalizable matrices, repeated eigenvalues have a full eigenspace and eigenvectors are stable.
  const gap = computeEigenvalueGap(eigenvalue, eigenvalues, index)
  if (gap < 1e-6 && !diagnostics.normal && diagnostics.diagonalizable === false) {
    severity = 'severe'
    issues.push(`Eigenvector sensitive to perturbations (eigenvalue gap ≈ ${gap.toExponential(2)})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 3) MILD: Representative eigenvector is non-orthogonal to others
  if (myRep && reps && reps.length > 1) {
    try {
      const others = reps.filter(r => r !== myRep)
      let maxDot = 0
      const myNorm = Math.sqrt(myRep.reduce((s, v) => s + v * v, 0))
      for (const other of others) {
        const otherNorm = Math.sqrt(other.reduce((s, v) => s + v * v, 0))
        if (myNorm > 1e-10 && otherNorm > 1e-10) {
          let dot = 0
          for (let i = 0; i < Math.min(myRep.length, other.length); i++) {
            dot += myRep[i] * other[i]
          }
          const cosAngle = Math.abs(dot / (myNorm * otherNorm))
          if (cosAngle > maxDot) maxDot = cosAngle
        }
      }
      if (maxDot > 0.1) { // threshold for "non-orthogonal"
        severity = 'mild'
        issues.push(`Representative eigenvector non-orthogonal to others (max cos θ ≈ ${maxDot.toFixed(3)})`)
        return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
      }
      // Note: If maxDot <= 0.1, orthogonality is acceptable - continue to check multiplicities
    } catch (e) {
      // ignore numeric failures - continue to multiplicity checks
    }
  }

  // Extract algebraic/geometric multiplicity values
  const algVal = (perEntry && (perEntry.algebraicMultiplicity !== undefined && perEntry.algebraicMultiplicity !== null)) ? perEntry.algebraicMultiplicity : (alg[index] !== undefined ? alg[index] : null)
  const dimVal = (perEntry && (perEntry.dimension !== undefined || perEntry.geometricMultiplicity !== undefined)) ? (perEntry.dimension ?? perEntry.geometricMultiplicity) : (geom[index] !== undefined ? geom[index] : null)

  // Check if eigenvalue is repeated AND has algebraic != geometric multiplicity
  const hasDefect = algVal && dimVal && algVal > 1 && algVal !== dimVal
  
  if (hasDefect) {
    // New severity calculation based on ρ = k/n where k is distinct eigenvalues, n is matrix dimension
    const uniqueEigenvalues = new Set()
    const tol = 1e-8
    for (const ev of eigenvalues) {
      let found = false
      for (const uev of uniqueEigenvalues) {
        if (Math.hypot((uev.real ?? 0) - (ev.real ?? 0), (uev.imag ?? 0) - (ev.imag ?? 0)) <= tol) {
          found = true
          break
        }
      }
      if (!found) uniqueEigenvalues.add(ev)
    }
    const k = uniqueEigenvalues.size
    const rho = k / n
    
    if (rho >= 0.7) {
      severity = 'safe'
      issues.push(`Repeated defective eigenvalue (alg=${algVal}, dim=${dimVal}) but high distinctness ratio (ρ=${rho.toFixed(2)})`)
    } else if (rho >= 0.3) {
      severity = 'severe'
      issues.push(`Repeated defective eigenvalue (alg=${algVal}, dim=${dimVal}) with moderate distinctness (ρ=${rho.toFixed(2)})`)
    } else {
      severity = 'critical'
      issues.push(`Repeated defective eigenvalue (alg=${algVal}, dim=${dimVal}) with low distinctness (ρ=${rho.toFixed(2)})`)
    }
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 5) Critical: high algebraic multiplicity WITH deficient geometric multiplicity
  // Only flag as critical if truly defective (geom < alg), not just for large multiplicity
  const highMultThreshold = Math.max(2, Math.ceil(0.75 * n))

  if (algVal && algVal >= highMultThreshold && dimVal !== null && dimVal < algVal) {
    severity = 'critical'
    issues.push(`High algebraic multiplicity (${algVal}) with deficient geometric multiplicity (${dimVal})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 2) Severe: repeated eigenvalue with deficient geometric multiplicity (defective)
  // Only flag if geom < alg; repeated eigenvalues with alg == geom are perfectly healthy
  if (algVal && algVal > 1 && dimVal !== null && dimVal < algVal) {
    severity = 'severe'
    issues.push(`Repeated eigenvalue with deficient geometry (alg=${algVal}, geo=${dimVal})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // Repeated eigenvalue with alg == geom is NOT a problem - it's diagonalizable
  // Don't flag healthy repeated eigenvalues as severe

  // 7) Moderate: eigenvalue approximately numerically equal to others (clustered)
  if (gap < 1e-6) {
    severity = 'moderate'
    issues.push(`Eigenvalue close to another (gap ≈ ${gap.toExponential(2)})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 8) Mild: this eigenvalue's alg==geom but not all eigenvalues satisfy alg==geom
  // Use perEigen data if available for more accurate comparison, otherwise fall back to arrays
  let allMatch = false
  if (perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
    // Check using perEigenInfo: each unique eigenvalue should have alg == geom
    allMatch = perEigen.every(p => {
      const pAlg = p?.algebraicMultiplicity
      const pGeom = p?.geometricMultiplicity ?? p?.dimension
      return pAlg !== undefined && pGeom !== undefined && pAlg === pGeom
    })
  } else {
    // Fallback to direct array comparison
    allMatch = (alg.length === geom.length) && alg.length > 0 && alg.every((a, i) => a === geom[i])
  }

  if (algVal !== null && dimVal !== null && algVal === dimVal && !allMatch) {
    severity = 'mild'
    issues.push('Algebraic == Geometric for this eigenvalue, but not all eigenvalues are healthy')
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 9) Safe: all eigenvalues have algebraic == geometric
  if (allMatch) {
    severity = 'safe'
    issues.push('All eigenvalues have algebraic == geometric multiplicities')
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // Fallback: neutral informational
  return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
}

// Helper functions

/**
 * Compute a simple determinant (exact for up to 3x3) as a stability heuristic.
 *
 * @param {number[][]} matrix
 * @returns {number}
 */
function computeDeterminant(matrix) {
  // Simple determinant for small matrices (up to 3x3)
  const n = matrix.length
  if (n === 0) return 0
  if (n === 1) return complexAbs(toComplex(matrix[0][0] || 0))
  if (n === 2) {
    const a = toComplex(matrix[0][0] || 0)
    const b = toComplex(matrix[0][1] || 0)
    const c = toComplex(matrix[1][0] || 0)
    const d = toComplex(matrix[1][1] || 0)
    return complexAbs(complexSub(complexMul(a, d), complexMul(b, c)))
  }
  if (n === 3) {
    const a = toComplex(matrix[0][0] || 0)
    const b = toComplex(matrix[0][1] || 0)
    const c = toComplex(matrix[0][2] || 0)
    const d = toComplex(matrix[1][0] || 0)
    const e = toComplex(matrix[1][1] || 0)
    const f = toComplex(matrix[1][2] || 0)
    const g = toComplex(matrix[2][0] || 0)
    const h = toComplex(matrix[2][1] || 0)
    const i = toComplex(matrix[2][2] || 0)
    const term1 = complexMul(a, complexSub(complexMul(e, i), complexMul(f, h)))
    const term2 = complexMul(b, complexSub(complexMul(d, i), complexMul(f, g)))
    const term3 = complexMul(c, complexSub(complexMul(d, h), complexMul(e, g)))
    return complexAbs(complexAdd(complexSub(term1, term2), term3))
  }
  
  // For larger matrices, estimate via product of diagonal (rough estimate)
  let prod = 1
  for (let i = 0; i < Math.min(n, matrix[0]?.length || 0); i++) {
    prod *= complexAbs(toComplex(matrix[i][i] || 0))
  }
  return prod
}

/**
 * Estimate condition number using row-sum ratios.
 *
 * @param {number[][]} matrix
 * @returns {number}
 */
function estimateConditionNumber(matrix) {
  // Rough estimate: ratio of max to min absolute row sums
  const rowNorms = matrix.map(row => row.reduce((sum, val) => sum + complexAbs(toComplex(val || 0)), 0))
  const maxNorm = Math.max(...rowNorms)
  const minNorm = Math.min(...rowNorms.filter(n => n > 0))
  return minNorm > 0 ? maxNorm / minNorm : Infinity
}

/**
 * Compute nearest-neighbor gaps for each eigenvalue.
 *
 * @param {Array<{real:number, imag:number}>} eigenvalues
 * @returns {number[]}
 */
function computeEigenvalueGaps(eigenvalues) {
  const gaps = []
  for (let i = 0; i < eigenvalues.length; i++) {
    gaps.push(computeEigenvalueGap(eigenvalues[i], eigenvalues, i))
  }
  return gaps
}

/**
 * Compute the minimum distance between one eigenvalue and the rest.
 *
 * @param {{real:number, imag:number}} eigenvalue
 * @param {Array<{real:number, imag:number}>} eigenvalues
 * @param {number} index
 * @returns {number}
 */
function computeEigenvalueGap(eigenvalue, eigenvalues, index) {
  let minGap = Infinity
  for (let j = 0; j < eigenvalues.length; j++) {
    if (j === index) continue
    const dist = Math.hypot(
      eigenvalue.real - eigenvalues[j].real,
      eigenvalue.imag - eigenvalues[j].imag
    )
    if (dist < minGap) minGap = dist
  }
  return minGap
}

/**
 * Group eigenvalues by proximity threshold.
 *
 * @param {Array<{real:number, imag:number}>} eigenvalues
 * @param {number} [threshold=1e-3]
 * @returns {number[][]}
 */
function findEigenvalueClusters(eigenvalues, threshold = 1e-3) {
  const clusters = []
  const visited = new Set()
  
  for (let i = 0; i < eigenvalues.length; i++) {
    if (visited.has(i)) continue
    const cluster = [i]
    visited.add(i)
    
    for (let j = i + 1; j < eigenvalues.length; j++) {
      if (visited.has(j)) continue
      const dist = Math.hypot(
        eigenvalues[i].real - eigenvalues[j].real,
        eigenvalues[i].imag - eigenvalues[j].imag
      )
      if (dist < threshold) {
        cluster.push(j)
        visited.add(j)
      }
    }
    
    if (cluster.length > 1) {
      clusters.push(cluster)
    }
  }
  
  return clusters
}

/**
 * Compute spread ratio of eigenvalue magnitudes.
 *
 * @param {Array<{real:number, imag:number}>} eigenvalues
 * @returns {number}
 */
function computeEigenvalueSpread(eigenvalues) {
  const magnitudes = eigenvalues.map(ev => Math.hypot(ev.real, ev.imag))
  const nonZero = magnitudes.filter(m => m > 1e-10)
  if (nonZero.length === 0) return 1
  const max = Math.max(...nonZero)
  const min = Math.min(...nonZero)
  return min > 0 ? max / min : Infinity
}

/**
 * Estimate orthogonality of matrix columns.
 *
 * @param {number[][]} matrix
 * @returns {number}
 */
function checkOrthogonality(matrix) {
  // Compute dot products between column vectors
  const n = matrix.length
  const m = matrix[0]?.length || 0
  
  if (n === 0 || m === 0) return 1
  
  let totalScore = 0
  let count = 0
  
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      let dot = { r: 0, i: 0 }
      let normI = 0
      let normJ = 0
      
      for (let k = 0; k < n; k++) {
        const vi = toComplex(matrix[k][i] || 0)
        const vj = toComplex(matrix[k][j] || 0)
        dot = complexAdd(dot, complexMul(complexConj(vi), vj))
        normI += complexAbs2(vi)
        normJ += complexAbs2(vj)
      }
      
      normI = Math.sqrt(normI)
      normJ = Math.sqrt(normJ)
      
      if (normI > 1e-10 && normJ > 1e-10) {
        const cosine = complexAbs(dot) / (normI * normJ)
        totalScore += (1 - cosine) // 1 if orthogonal, 0 if parallel
        count++
      }
    }
  }
  
  return count > 0 ? totalScore / count : 1
}

/**
 * Estimate the number of independent vectors using Gram-Schmidt.
 *
 * @param {number[][]} cols
 * @param {number} [tol=1e-8]
 * @returns {number}
 */
function estimateIndependentColumns(cols, tol = 1e-8) {
  if (!cols || cols.length === 0) return 0
  // ensure all columns have same length
  const n = Math.max(...cols.map(c => (c ? c.length : 0)))
  const normalizeCol = (c) => {
    const v = new Array(n).fill(null).map(() => ({ r: 0, i: 0 }))
    if (!c) return v
    for (let i = 0; i < c.length; i++) v[i] = toComplex(c[i] || 0)
    return v
  }

  const basis = []
  for (const col of cols) {
    const v = normalizeCol(col)
    let w = v.map(z => ({ r: z.r, i: z.i }))
    for (const b of basis) {
      // projection of w onto b
      let dotWB = { r: 0, i: 0 }
      let dotBB = { r: 0, i: 0 }
      for (let i = 0; i < n; i++) {
        dotWB = complexAdd(dotWB, complexMul(w[i], complexConj(b[i])))
        dotBB = complexAdd(dotBB, complexMul(b[i], complexConj(b[i])))
      }
      if (complexAbs(dotBB) > tol * tol) {
        const scale = complexDiv(dotWB, dotBB)
        for (let i = 0; i < n; i++) {
          w[i] = complexSub(w[i], complexMul(scale, b[i]))
        }
      }
    }
    let norm = 0
    for (let i = 0; i < n; i++) norm += complexAbs2(w[i])
    norm = Math.sqrt(norm)
    if (norm > tol) {
      // normalize and add to basis
      for (let i = 0; i < n; i++) w[i] = { r: w[i].r / norm, i: w[i].i / norm }
      basis.push(w)
    }
  }
  return basis.length
}

function normalizeMatrix(matrix) {
  if (!matrix) return null
  if (Array.isArray(matrix)) return matrix
  const data = matrix.data
  const imag = matrix.imag
  if (!Array.isArray(data)) return null
  if (imag && Array.isArray(imag)) {
    return data.map((row, rIdx) => row.map((val, cIdx) => ({
      real: val ?? 0,
      imag: imag?.[rIdx]?.[cIdx] ?? 0
    })))
  }
  return data
}

function toComplex(value) {
  if (value === null || value === undefined) return { r: 0, i: 0 }
  if (typeof value === 'number') return { r: value, i: 0 }
  if (typeof value === 'string') return parseComplexString(value)
  if (Array.isArray(value) && value.length === 2) return { r: Number(value[0]) || 0, i: Number(value[1]) || 0 }
  if (typeof value === 'object') {
    return { r: Number(value.real ?? value.r ?? 0) || 0, i: Number(value.imag ?? value.i ?? 0) || 0 }
  }
  return { r: 0, i: 0 }
}

function parseComplexString(input) {
  const s = String(input).trim()
  if (s === 'i' || s === '+i') return { r: 0, i: 1 }
  if (s === '-i') return { r: 0, i: -1 }
  if (s.toLowerCase().includes('i')) {
    const core = s.replace(/i$/i, '')
    let splitPos = -1
    for (let i = core.length - 1; i > 0; i--) {
      const ch = core[i]
      if (ch === '+' || ch === '-') { splitPos = i; break }
    }
    if (splitPos === -1) {
      const b = Number(core)
      return { r: 0, i: Number.isFinite(b) ? b : 0 }
    }
    const aStr = core.slice(0, splitPos)
    const bStr = core.slice(splitPos)
    const a = Number(aStr)
    const b = Number(bStr)
    return { r: Number.isFinite(a) ? a : 0, i: Number.isFinite(b) ? b : 0 }
  }
  const a = Number(s)
  return { r: Number.isFinite(a) ? a : 0, i: 0 }
}

function complexAdd(a, b) {
  return { r: a.r + b.r, i: a.i + b.i }
}

function complexSub(a, b) {
  return { r: a.r - b.r, i: a.i - b.i }
}

function complexMul(a, b) {
  return { r: a.r * b.r - a.i * b.i, i: a.r * b.i + a.i * b.r }
}

function complexDiv(a, b) {
  const denom = b.r * b.r + b.i * b.i
  if (denom === 0) return { r: 0, i: 0 }
  return { r: (a.r * b.r + a.i * b.i) / denom, i: (a.i * b.r - a.r * b.i) / denom }
}

function complexConj(a) {
  return { r: a.r, i: -a.i }
}

function complexAbs2(a) {
  return a.r * a.r + a.i * a.i
}

function complexAbs(a) {
  return Math.sqrt(complexAbs2(a))
}

/**
 * Compute which eigenvector indices are non-orthogonal to all other eigenvectors.
 * An eigenvector is considered non-orthogonal if it has |cos θ| > threshold with ANY other eigenvector.
 * 
 * @param {Object} diagnostics - The diagnostics object containing eigenvectors
 * @param {number} [threshold=0.1] - Threshold for non-orthogonality (|cos θ| > threshold)
 * @returns {Set<number>} Set of column indices that are non-orthogonal to all others
 */
export function computeNonOrthogonalEigenvectors(diagnostics, threshold = 0.1) {
  const nonOrthogonalIndices = new Set()
  
  if (!diagnostics) return nonOrthogonalIndices
  
  // Get eigenvector matrix
  const eigenvectors = normalizeMatrix(diagnostics.eigenvectors)
  if (!eigenvectors || eigenvectors.length === 0) return nonOrthogonalIndices
  
  const numRows = eigenvectors.length
  const numCols = eigenvectors[0]?.length || 0
  if (numCols === 0) return nonOrthogonalIndices
  
  // Extract each column as a vector
  const columns = []
  for (let c = 0; c < numCols; c++) {
    const col = []
    for (let r = 0; r < numRows; r++) {
      const val = eigenvectors[r][c]
      if (typeof val === 'object' && (val.real !== undefined || val.imag !== undefined)) {
        col.push({ r: val.real ?? 0, i: val.imag ?? 0 })
      } else if (typeof val === 'number') {
        col.push({ r: val, i: 0 })
      } else {
        col.push({ r: 0, i: 0 })
      }
    }
    columns.push(col)
  }
  
  // Compute norms
  const norms = columns.map(col => {
    let sum = 0
    for (const v of col) {
      sum += complexAbs2(v)
    }
    return Math.sqrt(sum)
  })
  
  // Check orthogonality between each pair
  for (let i = 0; i < numCols; i++) {
    if (norms[i] < 1e-10) continue // skip zero vectors
    
    let isOrthogonalToAll = true
    for (let j = 0; j < numCols; j++) {
      if (i === j || norms[j] < 1e-10) continue
      
      // Compute inner product <col_i, col_j> (conjugate linear in second arg)
      let innerProduct = { r: 0, i: 0 }
      for (let k = 0; k < numRows; k++) {
        // <u, v> = sum(u_k * conj(v_k))
        const prod = complexMul(columns[i][k], complexConj(columns[j][k]))
        innerProduct = complexAdd(innerProduct, prod)
      }
      
      const cosAngle = complexAbs(innerProduct) / (norms[i] * norms[j])
      if (cosAngle > threshold) {
        isOrthogonalToAll = false
        break
      }
    }
    
    if (!isOrthogonalToAll) {
      nonOrthogonalIndices.add(i)
    }
  }
  
  return nonOrthogonalIndices
}

