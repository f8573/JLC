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
  severe: 'Severe ❗',
  moderate: 'Moderate ⚠️',
  mild: 'Mild ℹ️',
  safe: 'Safe ✅'
}

/**
 * Compute overall spectral severity based on diagnostics
 */
export function computeSpectralSeverity(diagnostics) {
  if (!diagnostics) return { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }

  const issues = []
  let severity = 'safe'

  const eigenvalues = diagnostics.eigenvalues || []
  const eigenvectors = diagnostics.eigenvectors?.data
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
    const totalGeom = geom.reduce((sum, g) => sum + g, 0)
    const n = eigenvalues.length
    if (totalGeom < n) {
      severity = 'critical'
      issues.push(`Eigenbasis does not exist (${totalGeom} independent eigenvectors < ${n} dimension)`)
    }
  }

  // 1b. Representative eigenvectors for unique eigenvalues are not independent
  try {
    if (perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
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

  // 2. Geometric multiplicity much smaller than algebraic
  for (let i = 0; i < alg.length; i++) {
    if (alg[i] && geom[i] && geom[i] < alg[i] * 0.5) {
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
  // 1. Eigenvector sensitivity due to close eigenvalues
  const gaps = computeEigenvalueGaps(eigenvalues)
  for (let i = 0; i < gaps.length; i++) {
    if (gaps[i] < 1e-6 && !normal) {
      if (severity === 'safe' || severity === 'mild' || severity === 'moderate') {
        severity = 'severe'
      }
      issues.push(`Eigenvalue ${i + 1}: small gap (${gaps[i].toExponential(2)}), eigenvectors may be volatile`)
    }
  }

  // 2. Repeated eigenvalues with uncertain geometric multiplicity
  for (let i = 0; i < alg.length; i++) {
    if (alg[i] > 1 && geom[i] !== alg[i]) {
      if (severity === 'safe' || severity === 'mild' || severity === 'moderate') {
        severity = 'severe'
      }
      issues.push(`Eigenvalue ${i + 1}: repeated (alg=${alg[i]}) with uncertain geometry (geo=${geom[i]})`)
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
  // 1. Non-orthogonal eigenvectors
  if (!orthogonal && eigenvectors && eigenvectors.length > 0) {
    const orthoScore = checkOrthogonality(eigenvectors)
    if (orthoScore < 0.9) {
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
 * Compute per-eigenvalue severity
 */
export function computePerEigenvalueSeverity(eigenvalue, index, diagnostics) {
  if (!diagnostics) return { level: 'safe', color: SEVERITY_COLORS.safe, issues: [] }

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
    if (perEigen && Array.isArray(perEigen) && perEigen.length > 0) {
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

  // 2) SEVERE: Eigenvector sensitive to perturbations (small eigenvalue gap, non-normal matrix)
  const gap = computeEigenvalueGap(eigenvalue, eigenvalues, index)
  if (gap < 1e-6 && !diagnostics.normal) {
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
    } catch (e) {
      // ignore
    }
  }

  // 4) SAFE: Representative eigenvector is orthogonal to others
  if (myRep && reps && reps.length > 1) {
    severity = 'safe'
    issues.push('Representative eigenvector is orthogonal to others')
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
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

  // 5) Critical: algebraic multiplicity is high relative to matrix dimension
  // Treat "high multiplicity" as a large fraction of the matrix dimension.
  // Use 75% of n as the threshold so small matrices (n=3) treat multiplicity 2 as severe, not critical.
  const highMultThreshold = Math.max(2, Math.ceil(0.75 * n))

  if (algVal && algVal >= highMultThreshold) {
    severity = 'critical'
    issues.push(`High algebraic multiplicity (${algVal}) relative to dimension ${n}`)
    if (dimVal && dimVal < algVal) issues.push(`Geometric multiplicity (${dimVal}) < algebraic (${algVal})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 2) Severe: repeated eigenvalue (alg > 1) but not high multiplicity
  if (algVal && algVal > 1) {
    severity = 'severe'
    issues.push(`Repeated eigenvalue (algebraic multiplicity = ${algVal})`)
    if (dimVal && dimVal < algVal) issues.push(`Geometric multiplicity (${dimVal}) < algebraic (${algVal})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 7) Moderate: eigenvalue approximately numerically equal to others (clustered)
  if (gap < 1e-6) {
    severity = 'moderate'
    issues.push(`Eigenvalue close to another (gap ≈ ${gap.toExponential(2)})`)
    return { level: severity, color: SEVERITY_COLORS[severity], label: SEVERITY_LABELS[severity], issues }
  }

  // 8) Mild: this eigenvalue's alg==geom but not all eigenvalues satisfy alg==geom
  const allMatch = (alg.length === geom.length) && alg.every((a, i) => a === geom[i])

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

function computeDeterminant(matrix) {
  // Simple determinant for small matrices (up to 3x3)
  const n = matrix.length
  if (n === 0) return 0
  if (n === 1) return matrix[0][0] || 0
  if (n === 2) {
    return (matrix[0][0] || 0) * (matrix[1][1] || 0) - (matrix[0][1] || 0) * (matrix[1][0] || 0)
  }
  if (n === 3) {
    return (
      (matrix[0][0] || 0) * ((matrix[1][1] || 0) * (matrix[2][2] || 0) - (matrix[1][2] || 0) * (matrix[2][1] || 0)) -
      (matrix[0][1] || 0) * ((matrix[1][0] || 0) * (matrix[2][2] || 0) - (matrix[1][2] || 0) * (matrix[2][0] || 0)) +
      (matrix[0][2] || 0) * ((matrix[1][0] || 0) * (matrix[2][1] || 0) - (matrix[1][1] || 0) * (matrix[2][0] || 0))
    )
  }
  
  // For larger matrices, estimate via product of diagonal (rough estimate)
  let prod = 1
  for (let i = 0; i < Math.min(n, matrix[0]?.length || 0); i++) {
    prod *= matrix[i][i] || 0
  }
  return prod
}

function estimateConditionNumber(matrix) {
  // Rough estimate: ratio of max to min absolute row sums
  const rowNorms = matrix.map(row => row.reduce((sum, val) => sum + Math.abs(val || 0), 0))
  const maxNorm = Math.max(...rowNorms)
  const minNorm = Math.min(...rowNorms.filter(n => n > 0))
  return minNorm > 0 ? maxNorm / minNorm : Infinity
}

function computeEigenvalueGaps(eigenvalues) {
  const gaps = []
  for (let i = 0; i < eigenvalues.length; i++) {
    gaps.push(computeEigenvalueGap(eigenvalues[i], eigenvalues, i))
  }
  return gaps
}

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

function computeEigenvalueSpread(eigenvalues) {
  const magnitudes = eigenvalues.map(ev => Math.hypot(ev.real, ev.imag))
  const nonZero = magnitudes.filter(m => m > 1e-10)
  if (nonZero.length === 0) return 1
  const max = Math.max(...nonZero)
  const min = Math.min(...nonZero)
  return min > 0 ? max / min : Infinity
}

function checkOrthogonality(matrix) {
  // Compute dot products between column vectors
  const n = matrix.length
  const m = matrix[0]?.length || 0
  
  if (n === 0 || m === 0) return 1
  
  let totalScore = 0
  let count = 0
  
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < m; j++) {
      let dot = 0
      let normI = 0
      let normJ = 0
      
      for (let k = 0; k < n; k++) {
        const vi = matrix[k][i] || 0
        const vj = matrix[k][j] || 0
        dot += vi * vj
        normI += vi * vi
        normJ += vj * vj
      }
      
      normI = Math.sqrt(normI)
      normJ = Math.sqrt(normJ)
      
      if (normI > 1e-10 && normJ > 1e-10) {
        const cosine = Math.abs(dot / (normI * normJ))
        totalScore += (1 - cosine) // 1 if orthogonal, 0 if parallel
        count++
      }
    }
  }
  
  return count > 0 ? totalScore / count : 1
}

function estimateIndependentColumns(cols, tol = 1e-8) {
  if (!cols || cols.length === 0) return 0
  // ensure all columns have same length
  const n = Math.max(...cols.map(c => (c ? c.length : 0)))
  const normalizeCol = (c) => {
    const v = new Array(n).fill(0)
    if (!c) return v
    for (let i = 0; i < c.length; i++) v[i] = c[i] || 0
    return v
  }

  const basis = []
  for (const col of cols) {
    const v = normalizeCol(col)
    let w = v.slice()
    for (const b of basis) {
      // projection of w onto b
      let dotWB = 0, dotBB = 0
      for (let i = 0; i < n; i++) { dotWB += w[i] * b[i]; dotBB += b[i] * b[i] }
      if (dotBB > 0) {
        const scale = dotWB / dotBB
        for (let i = 0; i < n; i++) w[i] -= scale * b[i]
      }
    }
    let norm = 0
    for (let i = 0; i < n; i++) norm += w[i] * w[i]
    norm = Math.sqrt(norm)
    if (norm > tol) {
      // normalize and add to basis
      for (let i = 0; i < n; i++) w[i] = w[i] / norm
      basis.push(w)
    }
  }
  return basis.length
}
