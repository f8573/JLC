import React from 'react'
import { formatNumber } from '../../utils/format'

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
  if (!Array.isArray(data) || data.length === 0) {
    return (
      <div className={`text-xs text-slate-400 ${className}`}>
        No matrix data
      </div>
    )
  }

  const cols = data[0]?.length || 0
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(${cols}, minmax(${minCellWidth}px, 1fr))`,
    gap: `${gap}px`
  }

  return (
    <div className={className} style={gridStyle}>
      {data.flatMap((row, rIdx) =>
        row.map((value, cIdx) => (
          <div
            key={`m-${rIdx}-${cIdx}`}
            className={`${cellClassName} ${highlightDiagonal && rIdx === cIdx ? 'purple-glow' : ''}`.trim()}
          >
            {formatNumber(value, 2)}
          </div>
        ))
      )}
    </div>
  )
}

