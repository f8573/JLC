/**
 * Matrix operations utilities used by matrix analysis pages.
 */

/**
 * Extract numeric value from potentially complex entry.
 *
 * @param {any} value - Matrix entry (number, complex object, or array)
 * @returns {number} Numeric value (real part if complex)
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

/**
 * Get matrix data array from matrix object or raw array.
 *
 * @param {Object|number[][]} matrix - Matrix object with data property or raw 2D array
 * @returns {number[][]|null} 2D array of values
 */
export function getMatrixData(matrix) {
  if (!matrix) return null
  if (Array.isArray(matrix)) return matrix
  if (matrix.data && Array.isArray(matrix.data)) return matrix.data
  return null
}

/**
 * Multiply two matrices: C = A * B
 *
 * @param {number[][]} A - First matrix (m x n)
 * @param {number[][]} B - Second matrix (n x p)
 * @returns {number[][]|null} Result matrix (m x p)
 */
export function multiplyMatrices(A, B) {
  const dataA = getMatrixData(A)
  const dataB = getMatrixData(B)

  if (!dataA || !dataB) return null

  const m = dataA.length
  const n = dataA[0]?.length || 0
  const p = dataB[0]?.length || 0
  const n2 = dataB.length

  if (n !== n2 || n === 0 || p === 0) return null

  const result = []
  for (let i = 0; i < m; i++) {
    const row = []
    for (let j = 0; j < p; j++) {
      let sum = 0
      for (let k = 0; k < n; k++) {
        sum += getValue(dataA[i][k]) * getValue(dataB[k][j])
      }
      row.push(sum)
    }
    result.push(row)
  }
  return result
}

/**
 * Transpose a matrix.
 *
 * @param {number[][]} A - Matrix to transpose
 * @returns {number[][]|null} Transposed matrix
 */
export function transposeMatrix(A) {
  const data = getMatrixData(A)
  if (!data || data.length === 0) return null

  const m = data.length
  const n = data[0]?.length || 0

  const result = []
  for (let j = 0; j < n; j++) {
    const row = []
    for (let i = 0; i < m; i++) {
      row.push(getValue(data[i][j]))
    }
    result.push(row)
  }
  return result
}

/**
 * Create an identity matrix of size n.
 *
 * @param {number} n - Size of the identity matrix
 * @returns {number[][]} Identity matrix
 */
export function createIdentityMatrix(n) {
  const result = []
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0)
    row[i] = 1
    result.push(row)
  }
  return result
}

/**
 * Invert a numeric matrix using Gauss-Jordan elimination.
 * Returns null if the matrix is singular or invalid.
 * Operates on the numeric values extracted via `getValue`.
 *
 * @param {number[][]|Object} matrix
 * @returns {number[][]|null}
 */
export function invertMatrix(matrix) {
  const data = getMatrixData(matrix)
  if (!data || data.length === 0) return null
  const n = data.length
  if (data[0].length !== n) return null

  const A = new Array(n)
  for (let i = 0; i < n; i++) {
    A[i] = new Array(2 * n)
    for (let j = 0; j < n; j++) A[i][j] = getValue(data[i][j])
    for (let j = 0; j < n; j++) A[i][n + j] = (i === j) ? 1 : 0
  }

  for (let col = 0; col < n; col++) {
    let pivotRow = col
    for (let r = col; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivotRow][col])) pivotRow = r
    }
    if (Math.abs(A[pivotRow][col]) < 1e-14) return null

    if (pivotRow !== col) {
      const tmp = A[col]
      A[col] = A[pivotRow]
      A[pivotRow] = tmp
    }

    const pivot = A[col][col]
    for (let j = 0; j < 2 * n; j++) A[col][j] /= pivot

    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = A[r][col]
      if (factor === 0) continue
      for (let j = 0; j < 2 * n; j++) {
        A[r][j] -= factor * A[col][j]
      }
    }
  }

  const inv = new Array(n)
  for (let i = 0; i < n; i++) {
    inv[i] = new Array(n)
    for (let j = 0; j < n; j++) inv[i][j] = A[i][n + j]
  }
  return inv
}
