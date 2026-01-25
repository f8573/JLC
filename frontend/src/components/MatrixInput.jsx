import React from 'react'
import { useMatrix } from '../hooks/useMatrix'
import { useMatrixAnimation } from '../hooks/useMatrixAnimation'
import { analyzeAndCache, valuesToMatrixData, matrixToString } from '../utils/diagnostics'
import Badge from './ui/Badge'
import MatrixInputCard from './matrix/MatrixInputCard'
import MatrixDimensionControl from './matrix/MatrixDimensionControl'
import MatrixActions from './matrix/MatrixActions'
import FeatureGrid from './features/FeatureGrid'

export default function MatrixInput() {
  const { rows, cols, values, updateDimensions, updateCell, transpose } = useMatrix()
  const { containerRef, animateTranspose } = useMatrixAnimation()

  async function handleAnalyze() {
    const matrixData = valuesToMatrixData(values)
    const matrixString = matrixToString(matrixData)
    // store recent session (limit 10)
    try {
      const key = 'recentSessions'
      const raw = localStorage.getItem(key)
      const arr = raw ? JSON.parse(raw) : []
      arr.unshift({ title: matrixString, ts: Date.now() })
      const sliced = arr.slice(0, 10)
      localStorage.setItem(key, JSON.stringify(sliced))
    } catch (e) {
      // ignore
    }

    try {
      await analyzeAndCache(matrixData, matrixString)
    } catch (e) {
      // allow navigation even if diagnostics failed
    }

    window.location.href = '/matrix=' + encodeURIComponent(matrixString) + '/basic'
  }

  function handleTranspose() {
    animateTranspose(rows, cols, () => {
      transpose()
    })
  }

  function incRows() { updateDimensions(rows + 1, cols) }
  function decRows() { updateDimensions(Math.max(1, rows - 1), cols) }
  function incCols() { updateDimensions(rows, cols + 1) }
  function decCols() { updateDimensions(rows, Math.max(1, cols - 1)) }

  return (
    <main className="flex-1 overflow-y-auto bg-white math-grid relative flex flex-col items-center">
      <div className="w-full max-w-4xl px-8 pt-16 pb-24 flex flex-col items-center">
        <div className="text-center mb-12">
          <Badge animated>Computation Node 01</Badge>
          <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight mb-4 mt-6">
            Matrix Input Console
          </h1>
          <p className="text-slate-600 text-lg max-w-lg mx-auto">
            Define your matrix below to perform characteristic analysis and decomposition.
          </p>
        </div>
        
        <MatrixInputCard 
          matrixRef={containerRef}
          values={values}
          cols={cols}
          onCellChange={updateCell}
          onIncreaseRows={incRows}
          onDecreaseRows={decRows}
          onIncreaseCols={incCols}
          onDecreaseCols={decCols}
          onAnalyze={handleAnalyze}
        />
        
        <MatrixDimensionControl 
          rows={rows}
          cols={cols}
          onDimensionChange={updateDimensions}
        />
        
        <MatrixActions 
          onAnalyze={handleAnalyze}
          onTranspose={handleTranspose}
        />
        
        <FeatureGrid />
      </div>
    </main>
  )
}
