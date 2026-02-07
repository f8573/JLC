/**
 * Get the current precision settings from localStorage.
 * Returns { digits: number }
 */
function getGlobalPrecision() {
  const precision = localStorage.getItem('precision') || '6'
  const num = parseInt(precision, 10)
  return { digits: isNaN(num) ? 6 : num }
}

/**
 * Format a numeric value with fixed precision and trimmed zeros.
 * If digits is not provided, uses the global precision setting.
 *
 * @param {number} value
 * @param {number} [digits] - Number of decimal places. If omitted, uses global setting.
 * @returns {string}
 */
export function formatNumber(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (!Number.isFinite(value)) return String(value)

  // Use global precision if digits not specified
  const { digits: globalDigits } = getGlobalPrecision()
  const precision = digits !== undefined ? digits : globalDigits

  const fixed = Number(value).toFixed(precision)
  return fixed.replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1')
}

/**
 * Format a ratio as a percentage string.
 *
 * @param {number} value
 * @param {number} [digits=1]
 * @returns {string}
 */
export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const pct = Number(value) * 100
  return `${formatNumber(pct, digits)}%`
}

/**
 * Format matrix dimensions as "rows × cols".
 *
 * @param {number} rows
 * @param {number} cols
 * @returns {string}
 */
export function formatDimension(rows, cols) {
  if (!rows || !cols) return '—'
  return `${rows} × ${cols}`
}

/**
 * Format a complex number object as a string.
 * If digits is not provided, uses the global precision setting.
 *
 * @param {{real: number, imag: number}} value
 * @param {number} [digits] - Number of decimal places. If omitted, uses global setting.
 * @returns {string}
 */
export function formatComplex(value, digits) {
  if (!value) return '—'
  
  // Use global precision if digits not specified
  const { digits: globalDigits } = getGlobalPrecision()
  const precision = digits !== undefined ? digits : globalDigits
  
  const realNum = Number(value.real) || 0
  const imagNum = Number(value.imag) || 0
  const real = formatNumber(realNum, precision)
  const imagAbs = Math.abs(imagNum)

  // Pure real
  if (Math.abs(imagNum) < 1e-12) {
    return real
  }

  // Pure imaginary
  if (Math.abs(realNum) < 1e-12) {
    if (Math.abs(imagAbs - 1.0) < 1e-12) return imagNum < 0 ? '-i' : 'i'
    return imagNum < 0 ? `-${formatNumber(imagAbs, precision)}i` : `${formatNumber(imagAbs, precision)}i`
  }

  // Both real and imaginary present. Omit '1' coefficient.
  const sign = imagNum >= 0 ? '+' : '-'
  const imagCoeff = Math.abs(imagAbs - 1.0) < 1e-12 ? 'i' : `${formatNumber(imagAbs, precision)}i`
  return `${real}${sign}${imagCoeff}`
}

/**
 * Convert backend definiteness enum to a human-friendly label.
 *
 * @param {string|null|undefined} val
 * @returns {string}
 */
export function formatDefiniteness(val) {
  if (val === null || val === undefined) return 'Unknown'
  switch (val) {
    case 'POSITIVE_DEFINITE':
      return 'Positive Definite'
    case 'POSITIVE_SEMIDEFINITE':
      return 'Positive Semidefinite'
    case 'NEGATIVE_DEFINITE':
      return 'Negative Definite'
    case 'NEGATIVE_SEMIDEFINITE':
      return 'Negative Semidefinite'
    case 'INDEFINITE':
      return 'Indefinite'
    case 'UNDEFINED':
      return 'Undefined'
    default:
      // Fallback: prettify enum-like strings
      try {
        return String(val).replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase())
      } catch {
        return String(val)
      }
  }
}

