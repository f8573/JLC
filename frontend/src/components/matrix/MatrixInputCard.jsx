import React from 'react'
import MatrixGrid from './MatrixGrid'
import IconButton from '../ui/IconButton'

/**
 * Card wrapper for the matrix grid and dimension controls.
 *
 * @param {Object} props
 * @param {React.RefObject<HTMLDivElement>} props.matrixRef
 * @param {string[][]} props.values
 * @param {number} props.cols
 * @param {(row: number, col: number, value: string) => void} props.onCellChange
 * @param {() => void} props.onIncreaseRows
 * @param {() => void} props.onDecreaseRows
 * @param {() => void} props.onIncreaseCols
 * @param {() => void} props.onDecreaseCols
 * @param {() => void} [props.onAnalyze]
 */
export default function MatrixInputCard({ 
  matrixRef, 
  values, 
  cols, 
  onCellChange,
  onIncreaseRows,
  onDecreaseRows,
  onIncreaseCols,
  onDecreaseCols,
  onAnalyze
}) {
  return (
    <div className="w-full bg-white dark:bg-slate-800 border border-border-color dark:border-slate-700 rounded-xl p-10 md:p-16 shadow-xl relative overflow-hidden mb-10 transition-colors duration-300">
      <div className="absolute top-0 right-0 p-4">
        <span className="text-[10px] font-mono text-slate-300 dark:text-slate-600 select-none">
          ID: MATH-MX-842
        </span>
      </div>
      <div className="flex flex-col items-center justify-center">
        <div className="flex items-center gap-6 mb-12 relative">
          <span className="font-mono text-5xl text-primary font-light italic">A = </span>
          <div 
            ref={matrixRef} 
            className="matrix-bracket px-6 pt-6 pb-12 flex bg-slate-50 dark:bg-slate-700/50 rounded-sm relative transition-colors duration-300" 
            style={{ minWidth: cols * 90 }}
          >
            <MatrixGrid 
              values={values} 
              cols={cols} 
              onCellChange={onCellChange}
              onAnalyze={onAnalyze}
            />
            <div className="flex flex-col justify-center gap-3 ml-4">
              <IconButton icon="add" variant="small" onClick={onIncreaseRows} />
              <IconButton icon="remove" variant="small" onClick={onDecreaseRows} />
            </div>
            <div className="absolute bottom-2 left-6 right-12 flex justify-center gap-3">
              <IconButton icon="add" variant="small" onClick={onIncreaseCols} />
              <IconButton icon="remove" variant="small" onClick={onDecreaseCols} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
