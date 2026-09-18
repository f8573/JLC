import React from 'react'
import { formatComplex, formatNumber } from '../../utils/format'
import { usePrecisionUpdate } from '../../hooks/usePrecisionUpdate'

/**
 * Render a numeric matrix as a CSS grid.
 *
 * @param {Object} props
 * @param {number[][]} props.data
 * @param {number} [props.minCellWidth=60]
 * @param {number} [props.gap=8]
 * @param {string} [props.className='']
 * @param {string} [props.cellClassName='']
 * @param {boolean} [props.highlightDiagonal=false]
 */
export default function MatrixDisplay({
  data,
  minCellWidth = 60,
  gap = 8,
  className = '',
  cellClassName = '',
  highlightDiagonal = false
}) {
  // Subscribe to precision changes to trigger re-render
  usePrecisionUpdate()
  const source = data
  const baseData = source?.data || source
  const imag = source?.imag || null

  if (!Array.isArray(baseData) || baseData.length === 0) {
    return (
      <div className={`text-xs text-slate-400 ${className}`}>
        No matrix data
      </div>
    )
  }

  const cols = baseData[0]?.length || 0
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(${cols}, minmax(${minCellWidth}px, 1fr))`,
    gap: `${gap}px`
  }

  const renderValue = (value, rIdx, cIdx) => {
    if (imag && Array.isArray(imag) && Array.isArray(imag[rIdx])) {
      const real = Number(value ?? 0)
      const imagVal = Number(imag?.[rIdx]?.[cIdx] ?? 0)
      return formatComplex({ real, imag: imagVal })
    }
    if (value === null || value === undefined) return '—'
    if (typeof value === 'string') return value
    if (Array.isArray(value) && value.length === 2) {
      const real = Number(value[0])
      const imag = Number(value[1])
      return formatComplex({ real, imag })
    }
    if (typeof value === 'object' && (value.real !== undefined || value.imag !== undefined)) {
      const real = Number(value.real ?? value.r ?? 0)
      const imag = Number(value.imag ?? value.i ?? 0)
      return formatComplex({ real, imag })
    }
    return formatNumber(value)
  }

  return (
    <div className={className} style={gridStyle}>
      {baseData.flatMap((row, rIdx) =>
        row.map((value, cIdx) => (
          <div
            key={`m-${rIdx}-${cIdx}`}
            className={`${cellClassName} ${highlightDiagonal && rIdx === cIdx ? 'purple-glow' : ''}`.trim()}
          >
            {renderValue(value, rIdx, cIdx)}
          </div>
        ))
      )}
    </div>
  )
}
