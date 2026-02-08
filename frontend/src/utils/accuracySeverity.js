/**
 * Accuracy severity classification for matrix decomposition validation.
 * Maps Java AccuracyLevel enum values to frontend severity display.
 *
 * AccuracyLevel mapping:
 *   EXCELLENT → safe
 *   GOOD      → safe
 *   ACCEPTABLE → mild
 *   WARNING   → moderate
 *   POOR      → severe
 *   CRITICAL  → critical
 */

export const ACCURACY_COLORS = {
  critical: '#D32F2F',    // critical/dangerous
  severe: '#F57C00',      // severe/unstable
  moderate: '#FBC02D',    // moderate/caution
  mild: '#1976D2',        // mild/informational
  safe: '#2F6B3C'         // healthy/ideal (muted)
}

const ACCURACY_LABELS = {
  critical: 'Critical 🚨',
  severe: 'Poor ✴️',
  moderate: 'Warning ⚠️',
  mild: 'Acceptable ℹ️',
  safe: 'Good ✅'
}

const ACCURACY_SHORT_LABELS = {
  critical: '🚨',
  severe: '✴️',
  moderate: '⚠️',
  mild: 'ℹ️',
  safe: '✅'
}

/**
 * Map AccuracyLevel from Java API to frontend severity level.
 *
 * @param {string} level - AccuracyLevel name (EXCELLENT, GOOD, ACCEPTABLE, WARNING, POOR, CRITICAL)
 * @returns {string} Frontend severity level (safe, mild, moderate, severe, critical)
 */
function mapAccuracyLevel(level) {
  if (!level) return 'safe'
  
  const normalized = level.toUpperCase()
  switch (normalized) {
    case 'EXCELLENT':
    case 'GOOD':
      return 'safe'
    case 'ACCEPTABLE':
      return 'mild'
    case 'WARNING':
      return 'moderate'
    case 'POOR':
      return 'severe'
    case 'CRITICAL':
      return 'critical'
    default:
      return 'safe'
  }
}

/**
 * Get the worst severity level between norm and element levels.
 *
 * @param {string} normLevel - Norm-based AccuracyLevel
 * @param {string} elementLevel - Element-based AccuracyLevel
 * @returns {string} Worst severity level
 */
function getWorstLevel(normLevel, elementLevel) {
  const severityOrder = ['safe', 'mild', 'moderate', 'severe', 'critical']
  const normSeverity = mapAccuracyLevel(normLevel)
  const elemSeverity = mapAccuracyLevel(elementLevel)
  
  const normIdx = severityOrder.indexOf(normSeverity)
  const elemIdx = severityOrder.indexOf(elemSeverity)
  
  return severityOrder[Math.max(normIdx, elemIdx)]
}

/**
 * Compute accuracy severity from validation data.
 *
 * @param {Object} validation - Validation object from API
 * @param {string} validation.normLevel - Norm-based AccuracyLevel
 * @param {string} validation.elementLevel - Element-based AccuracyLevel
 * @param {number} validation.normResidual - Frobenius norm residual
 * @param {number} validation.elementResidual - Max element-wise residual
 * @param {string} validation.message - Diagnostic message
 * @param {boolean} validation.passes - Whether validation passes
 * @param {boolean} validation.shouldWarn - Whether to show warning
 * @returns {{level: string, color: string, label: string, shortLabel: string, normLevel: string, elementLevel: string}}
 */
export function computeAccuracySeverity(validation) {
  if (!validation) {
    return {
      level: 'safe',
      color: ACCURACY_COLORS.safe,
      label: ACCURACY_LABELS.safe,
      shortLabel: ACCURACY_SHORT_LABELS.safe,
      normLevel: 'safe',
      elementLevel: 'safe'
    }
  }

  const normSeverity = mapAccuracyLevel(validation.normLevel)
  const elemSeverity = mapAccuracyLevel(validation.elementLevel)
  const worstLevel = getWorstLevel(validation.normLevel, validation.elementLevel)

  return {
    level: worstLevel,
    color: ACCURACY_COLORS[worstLevel],
    label: ACCURACY_LABELS[worstLevel],
    shortLabel: ACCURACY_SHORT_LABELS[worstLevel],
    normLevel: normSeverity,
    elementLevel: elemSeverity
  }
}

/**
 * Compute per-element severity for highlighting in residual matrix.
 * Classifies individual element errors using the same threshold logic as Java.
 *
 * @param {number} elementError - Absolute or relative error for this element
 * @param {number} n - Matrix dimension
 * @param {number} [conditionNumber=1] - Estimated condition number
 * @returns {string} Severity level for this element
 */
function computeElementSeverity(elementError, n, conditionNumber = 1) {
  const EPS = 2.220446049250313e-16
  const base = n * EPS
  const condFactor = Math.max(1.0, Math.log10(Math.max(conditionNumber, 1.0)))

  const thresholds = [
    base * 10,                        // Excellent
    base * 100 * condFactor,          // Good
    base * 1000 * condFactor,         // Acceptable
    base * 10000 * condFactor,        // Warning
    base * 100000 * condFactor        // Poor (above = critical)
  ]

  if (elementError <= thresholds[0]) return 'safe'
  if (elementError <= thresholds[1]) return 'safe'
  if (elementError <= thresholds[2]) return 'mild'
  if (elementError <= thresholds[3]) return 'moderate'
  if (elementError <= thresholds[4]) return 'severe'
  return 'critical'
}

/**
 * Compute the residual matrix R = A - A' where A' is the reconstruction.
 *
 * @param {number[][]} original - Original matrix A
 * @param {number[][]} reconstructed - Reconstructed matrix A'
 * @returns {number[][]} Residual matrix
 */
export function computeResidualMatrix(original, reconstructed) {
  if (!original || !reconstructed) return null
  
  const rows = original.length
  const cols = original[0]?.length || 0
  
  const residual = []
  for (let i = 0; i < rows; i++) {
    const row = []
    for (let j = 0; j < cols; j++) {
      const origVal = getValue(original[i]?.[j])
      const reconVal = getValue(reconstructed[i]?.[j])
      row.push(origVal - reconVal)
    }
    residual.push(row)
  }
  return residual
}

/**
 * Compute per-element relative errors for coloring.
 * Uses the same logic as the backend to avoid spurious errors for near-zero elements.
 *
 * @param {number[][]} original - Original matrix A
 * @param {number[][]} reconstructed - Reconstructed matrix A'
 * @returns {{errors: number[][], severities: string[][]}} Element errors and severities
 */
export function computeElementErrors(original, reconstructed) {
  if (!original || !reconstructed) return { errors: null, severities: null }
  
  const rows = original.length
  const cols = original[0]?.length || 0
  const n = Math.max(rows, cols)
  
  // Threshold for considering an element "effectively zero"
  // Same as backend to ensure consistent results
  const ZERO_THRESHOLD = 1e-12
  
  const errors = []
  const severities = []
  
  for (let i = 0; i < rows; i++) {
    const errorRow = []
    const severityRow = []
    for (let j = 0; j < cols; j++) {
      const origVal = getValue(original[i]?.[j])
      const reconVal = getValue(reconstructed[i]?.[j])
      const absError = Math.abs(origVal - reconVal)
      
      // Compute relative error with appropriate scaling
      // If the original value is effectively zero, use absolute error
      // to avoid spurious large relative errors
      let relError
      if (Math.abs(origVal) < ZERO_THRESHOLD) {
        // Original is ~0: use absolute error only
        relError = absError
      } else {
        // Use relative error scaled by the original value
        relError = absError / Math.abs(origVal)
      }
      
      errorRow.push(relError)
      severityRow.push(computeElementSeverity(relError, n))
    }
    errors.push(errorRow)
    severities.push(severityRow)
  }
  
  return { errors, severities }
}

/**
 * Extract numeric value from potentially complex entry.
 *
 * @param {any} value - Matrix entry (number, complex object, or array)
 * @returns {number} Numeric value
 */
function getValue(value) {
  if (value === null || value === undefined) return 0
  if (typeof value === 'number') return value
  if (Array.isArray(value) && value.length >= 1) return Number(value[0]) || 0
  if (typeof value === 'object') {
    return Number(value.real ?? value.r ?? 0) || 0
  }
  return Number(value) || 0
}
