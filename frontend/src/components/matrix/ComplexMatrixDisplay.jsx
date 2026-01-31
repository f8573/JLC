import React from 'react'
import { formatNumber } from '../../utils/format'

/**
 * Render a complex-valued matrix where each entry is shown as `a + bi`.
 * Accepts matrix in the form { data: number[][], imag?: number[][] } or a plain 2D array
 */
export default function ComplexMatrixDisplay({ matrix, minCellWidth = 60, gap = 8, className = '', cellClassName = '' }) {
  if (!matrix) {
    return <div className={`text-xs text-slate-400 ${className}`.trim()}>No matrix data</div>
  }

  const data = matrix.data || matrix
  const imag = matrix.imag || null
  if (!Array.isArray(data) || data.length === 0) {
    return <div className={`text-xs text-slate-400 ${className}`.trim()}>No matrix data</div>
  }

  const rows = data.length
  const cols = data[0]?.length || 0

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(${cols}, minmax(${minCellWidth}px, 1fr))`,
    gap: `${gap}px`
  }

  const eps = 1e-12

  const formatEntry = (reRaw, imRaw) => {
    const re = Number(reRaw) || 0
    const im = Number(imRaw) || 0
    const isReZero = Math.abs(re) < eps
    const isImZero = Math.abs(im) < eps
    const a = formatNumber(re, 3)
    const bAbs = Math.abs(im)
    const b = formatNumber(bAbs, 3)

    // Pure real
    if (isImZero) {
      return a
    }

    // Pure imaginary
    if (isReZero) {
      if (Math.abs(bAbs - 1.0) < eps) {
        return im < 0 ? '-i' : 'i'
      }
      return im < 0 ? `-${b}i` : `${b}i`
    }

    // Both non-zero: a+bi, a-bi, -a+bi, -a-bi. Omit coefficient when |b| == 1
    const sign = im >= 0 ? '+' : '-'
    const imagCoeff = (Math.abs(bAbs - 1.0) < eps) ? 'i' : `${b}i`
    return `${a}${sign}${imagCoeff}`
  }

  return (
    <div className={className} style={gridStyle}>
      {data.flatMap((row, rIdx) =>
        row.map((val, cIdx) => {
          const re = val ?? 0
          const imv = imag && imag[rIdx] ? (imag[rIdx][cIdx] ?? 0) : 0
          return (
            <div key={`cm-${rIdx}-${cIdx}`} className={cellClassName}>
              {formatEntry(re, imv)}
            </div>
          )
        })
      )}
    </div>
  )
}
