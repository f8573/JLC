import React from 'react'
import Latex from '../ui/Latex'
import { formatComplex, formatNumber } from '../../utils/format'

/**
 * Render a numeric/complex matrix as LaTeX bmatrix.
 *
 * @param {Object} props
 * @param {number[][] | string[][] | Object} props.data
 * @param {Object} [props.matrix]
 * @param {boolean} [props.displayMode=false]
 * @param {string} [props.className='']
 * @param {Set<number>} [props.highlightColumns] - Set of column indices to highlight in blue (for non-orthogonal eigenvectors)
 */
export default function MatrixLatex({ data, matrix, displayMode = false, className = '', highlightColumns = null }) {
  const source = matrix ?? data
  if (!source) {
    return <div className={`text-xs text-slate-400 ${className}`.trim()}>No matrix data</div>
  }

  const baseData = source.data || source
  const imag = source.imag || null

  if (!Array.isArray(baseData) || baseData.length === 0) {
    return <div className={`text-xs text-slate-400 ${className}`.trim()}>No matrix data</div>
  }

  const formatEntry = (value, rIdx, cIdx) => {
    if (imag && Array.isArray(imag) && Array.isArray(imag[rIdx])) {
      const real = Number(value ?? 0)
      const imagVal = Number(imag?.[rIdx]?.[cIdx] ?? 0)
      return formatComplex({ real, imag: imagVal }, 2)
    }
    if (value === null || value === undefined) return '\\text{—}'
    if (typeof value === 'string') return value
    if (Array.isArray(value) && value.length === 2) {
      const real = Number(value[0])
      const imagVal = Number(value[1])
      return formatComplex({ real, imag: imagVal }, 2)
    }
    if (typeof value === 'object' && (value.real !== undefined || value.imag !== undefined || value.r !== undefined || value.i !== undefined)) {
      const real = Number(value.real ?? value.r ?? 0)
      const imagVal = Number(value.imag ?? value.i ?? 0)
      return formatComplex({ real, imag: imagVal }, 2)
    }
    return formatNumber(value, 2)
  }

  const rows = baseData.map((row, rIdx) => {
    if (!Array.isArray(row)) return ''
    const cols = row.map((value, cIdx) => {
      const formatted = formatEntry(value, rIdx, cIdx)
      // Highlight columns in blue if they are in the highlightColumns set (non-orthogonal eigenvectors)
      if (highlightColumns && highlightColumns.has(cIdx)) {
        return `{\\color{blue}${formatted}}`
      }
      return formatted
    }).join(' & ')
    return cols
  })

  const tex = `\\begin{bmatrix}${rows.join('\\\\')}\\end{bmatrix}`

  return <Latex tex={tex} displayMode={displayMode} className={className} />
}
