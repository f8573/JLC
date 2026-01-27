import React from 'react'
import MatrixCell from './MatrixCell'

/**
 * Grid of matrix input cells.
 *
 * @param {Object} props
 * @param {string[][]} props.values
 * @param {number} props.cols
 * @param {(row: number, col: number, value: string) => void} props.onCellChange
 * @param {() => void} [props.onAnalyze]
 */
export default function MatrixGrid({ values, cols, onCellChange, onAnalyze }) {
  return (
    <div 
      className="grid" 
      style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 80px)`, gap: '16px' }}
    >
      {values.map((row, rIdx) => 
        row.map((cell, cIdx) => (
          <MatrixCell
            key={`r${rIdx}c${cIdx}`}
            value={cell}
            rowIndex={rIdx}
            colIndex={cIdx}
            rows={values.length}
            cols={cols}
            onChange={onCellChange}
            onAnalyze={onAnalyze}
          />
        ))
      )}
    </div>
  )
}
