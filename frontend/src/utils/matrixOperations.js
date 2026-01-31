/**
 * Matrix operations utilities for computing reconstructions.
 */

/**
 * Extract numeric value from potentially complex entry.
 *
 * @param {any} value - Matrix entry (number, complex object, or array)
 * @returns {number} Numeric value (real part if complex)
 */
export function getValue(value) {
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
 * Compute QR reconstruction: A = Q * R
 *
 * @param {Object} qr - QR decomposition object with q and r properties
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructQR(qr) {
  if (!qr?.q?.data || !qr?.r?.data) return null
  return multiplyMatrices(qr.q.data, qr.r.data)
}

/**
 * Compute LU reconstruction: A = P^{-1} * L * U = P^T * L * U (since PA = LU)
 * For permutation matrices, P^{-1} = P^T.
 *
 * @param {Object} lu - LU decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructLU(lu) {
  if (!lu?.l?.data || !lu?.u?.data) return null
  const LU = multiplyMatrices(lu.l.data, lu.u.data)
  if (lu.p?.data) {
    // P^{-1} * L * U = P^T * L * U (P is orthogonal)
    return multiplyMatrices(transposeMatrix(lu.p.data), LU)
  }
  return LU
}

/**
 * Compute Cholesky reconstruction: A = L * L^T
 *
 * @param {Object} chol - Cholesky decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructCholesky(chol) {
  if (!chol?.l?.data) return null
  const L = chol.l.data
  const LT = transposeMatrix(L)
  return multiplyMatrices(L, LT)
}

/**
 * Compute SVD reconstruction: A = U * Σ * V^T
 *
 * @param {Object} svd - SVD decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructSVD(svd) {
  if (!svd?.u?.data || !svd?.sigma?.data || !svd?.v?.data) return null
  const US = multiplyMatrices(svd.u.data, svd.sigma.data)
  const VT = transposeMatrix(svd.v.data)
  return multiplyMatrices(US, VT)
}

/**
 * Compute Polar reconstruction: A = U * H
 *
 * @param {Object} polar - Polar decomposition object (u and p/h)
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructPolar(polar) {
  if (!polar?.u?.data || !polar?.p?.data) return null
  return multiplyMatrices(polar.u.data, polar.p.data)
}

/**
 * Compute Hessenberg reconstruction: A = Q * H * Q^T
 *
 * @param {Object} hess - Hessenberg decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructHessenberg(hess) {
  if (!hess?.q?.data || !hess?.h?.data) return null
  const QH = multiplyMatrices(hess.q.data, hess.h.data)
  const QT = transposeMatrix(hess.q.data)
  return multiplyMatrices(QH, QT)
}

/**
 * Compute Schur reconstruction: A = U * T * U^T
 *
 * @param {Object} schur - Schur decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructSchur(schur) {
  if (!schur?.u?.data || !schur?.t?.data) return null
  const UT = multiplyMatrices(schur.u.data, schur.t.data)
  const UH = transposeMatrix(schur.u.data)
  return multiplyMatrices(UT, UH)
}

/**
 * Compute Eigendecomposition reconstruction: A = P * D * P^(-1)
 *
 * @param {Object} diag - Diagonalization object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructEigen(diag) {
  if (!diag?.p?.data || !diag?.d?.data || !diag?.pInverse?.data) return null
  const PD = multiplyMatrices(diag.p.data, diag.d.data)
  return multiplyMatrices(PD, diag.pInverse.data)
}

/**
 * Compute Symmetric Spectral reconstruction: A = Q * Λ * Q^T
 *
 * @param {Object} spectral - Symmetric spectral decomposition object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructSymmetricSpectral(spectral) {
  if (!spectral?.q?.data || !spectral?.lambda?.data) return null
  const QL = multiplyMatrices(spectral.q.data, spectral.lambda.data)
  const QT = transposeMatrix(spectral.q.data)
  return multiplyMatrices(QL, QT)
}

/**
 * Compute Bidiagonalization reconstruction: A = U * B * V^T
 *
 * @param {Object} bidiag - Bidiagonalization object
 * @returns {number[][]|null} Reconstructed matrix
 */
export function reconstructBidiagonal(bidiag) {
  if (!bidiag?.u?.data || !bidiag?.b?.data || !bidiag?.v?.data) return null
  const UB = multiplyMatrices(bidiag.u.data, bidiag.b.data)
  const VT = transposeMatrix(bidiag.v.data)
  return multiplyMatrices(UB, VT)
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

  // build augmented matrix [A | I]
  const A = new Array(n)
  for (let i = 0; i < n; i++) {
    A[i] = new Array(2 * n)
    for (let j = 0; j < n; j++) A[i][j] = getValue(data[i][j])
    for (let j = 0; j < n; j++) A[i][n + j] = (i === j) ? 1 : 0
  }

  // Gauss-Jordan elimination
  for (let col = 0; col < n; col++) {
    // find pivot
    let pivotRow = col
    for (let r = col; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivotRow][col])) pivotRow = r
    }
    if (Math.abs(A[pivotRow][col]) < 1e-14) return null // singular

    // swap
    if (pivotRow !== col) {
      const tmp = A[col]; A[col] = A[pivotRow]; A[pivotRow] = tmp
    }

    // normalize pivot row
    const pivot = A[col][col]
    for (let j = 0; j < 2 * n; j++) A[col][j] /= pivot

    // eliminate other rows
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = A[r][col]
      if (factor === 0) continue
      for (let j = 0; j < 2 * n; j++) {
        A[r][j] -= factor * A[col][j]
      }
    }
  }

  // extract inverse
  const inv = new Array(n)
  for (let i = 0; i < n; i++) {
    inv[i] = new Array(n)
    for (let j = 0; j < n; j++) inv[i][j] = A[i][n + j]
  }
  return inv
}
/**
 * Compute inverse validation reconstruction: A * A^{-1}
 * This should produce the identity matrix I.
 *
 * @param {number[][]} original - Original matrix A
 * @param {number[][]} inverse - Inverse matrix A^{-1}
 * @returns {number[][]|null} Product A * A^{-1} which should be ≈ I
 */
export function reconstructInverseProduct(original, inverse) {
  if (!original || !inverse) return null
  return multiplyMatrices(original, inverse)
}
