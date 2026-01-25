export function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (!Number.isFinite(value)) return String(value)
  const fixed = Number(value).toFixed(digits)
  return fixed.replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1')
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const pct = Number(value) * 100
  return `${formatNumber(pct, digits)}%`
}

export function formatDimension(rows, cols) {
  if (!rows || !cols) return '—'
  return `${rows} × ${cols}`
}

export function formatComplex(value, digits = 3) {
  if (!value) return '—'
  const real = formatNumber(value.real, digits)
  const imag = formatNumber(value.imag, digits)
  const imagNum = Number(value.imag)
  if (!imagNum || Math.abs(imagNum) < 1e-12) {
    return real
  }
  const sign = imagNum >= 0 ? '+' : '-'
  const imagAbs = formatNumber(Math.abs(imagNum), digits)
  return `${real} ${sign} ${imagAbs}i`
}

export function formatMatrixSummary(matrix) {
  if (!Array.isArray(matrix)) return '—'
  const rows = matrix.length
  const cols = matrix[0]?.length || 0
  return `${rows}x${cols}`
}

