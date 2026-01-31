import React from 'react'
import FavoriteModal from './ui/FavoriteModal'
import { useMatrix } from '../hooks/useMatrix'
import { useMatrixAnimation } from '../hooks/useMatrixAnimation'
import { analyzeAndCache, valuesToMatrixData, matrixToString } from '../utils/diagnostics'
import Badge from './ui/Badge'
import MatrixInputCard from './matrix/MatrixInputCard'
import MatrixDimensionControl from './matrix/MatrixDimensionControl'
import MatrixActions from './matrix/MatrixActions'
import FeatureGrid from './features/FeatureGrid'

/**
 * Interactive matrix input console with dimension controls and analysis actions.
 */
export default function MatrixInput() {
  const { rows, cols, values, updateDimensions, updateCell, transpose } = useMatrix()
  const { containerRef, animateTranspose } = useMatrixAnimation()
  const [favoriteModalOpen, setFavoriteModalOpen] = React.useState(false)
  const [favoriteDefaultName, setFavoriteDefaultName] = React.useState('')
  const [savedMessage, setSavedMessage] = React.useState('')

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

  function handleFavorite() {
    const matrixData = valuesToMatrixData(values)
    const matrixString = matrixToString(matrixData)
    setFavoriteDefaultName(matrixString.slice(0, 40))
    // store matrixString temporarily on window to access in saver
    window.__pendingFavoriteMatrix = { matrixData, matrixString }
    setFavoriteModalOpen(true)
  }

  function saveFavorite(name) {
    const pending = window.__pendingFavoriteMatrix
    if (!pending) return setFavoriteModalOpen(false)
    const matrixData = pending.matrixData
    const matrixString = pending.matrixString

    // compute simple metadata
    const r = matrixData.rows || rows
    const c = matrixData.cols || cols
    let nonzero = 0
    for (let i = 0; i < r; ++i) {
      for (let j = 0; j < c; ++j) {
        const v = ((matrixData.data && matrixData.data[i] && matrixData.data[i][j]) ?? (values[i] && values[i][j]))
        if (v !== 0 && v !== '0' && v !== null && v !== '' && v !== undefined) nonzero++
      }
    }
    const density = r * c ? nonzero / (r * c) : 0

    // heuristics for type
    let type = 'general'
    // identity check
    let isIdentity = true
    for (let i = 0; i < r; ++i) {
      for (let j = 0; j < c; ++j) {
        const v = matrixData.data && matrixData.data[i] && matrixData.data[i][j]
        const expected = i === j ? 1 : 0
        if (v != null && Number(v) !== expected) { isIdentity = false; break }
      }
      if (!isIdentity) break
    }
    if (isIdentity) type = 'identity'

    // symmetric check (square only)
    if (type === 'general' && r === c) {
      let isSym = true
      for (let i = 0; i < r; ++i) {
        for (let j = 0; j < c; ++j) {
          const a = matrixData.data && matrixData.data[i] && matrixData.data[i][j]
          const b = matrixData.data && matrixData.data[j] && matrixData.data[j][i]
          if ((a || 0) !== (b || 0)) { isSym = false; break }
        }
        if (!isSym) break
      }
      if (isSym) type = 'symmetric'
    }

    try {
      const key = 'favorites'
      const raw = localStorage.getItem(key)
      const arr = raw ? JSON.parse(raw) : []
      const existing = arr.findIndex(item => item.matrixString === matrixString)
      const entry = { name, matrixString, rows: r, cols: c, density, type, ts: Date.now() }
      if (existing >= 0) {
        // update existing favorite's name and metadata
        arr[existing] = { ...arr[existing], ...entry }
      } else {
        arr.unshift(entry)
      }
      // ensure uniqueness by matrixString
      const deduped = []
      for (const it of arr) {
        if (!deduped.find(d => d.matrixString === it.matrixString)) deduped.push(it)
      }
      localStorage.setItem(key, JSON.stringify(deduped))
      setSavedMessage('Saved to favorites')
      setTimeout(() => setSavedMessage(''), 2500)
    } catch (e) {
      // ignore
    }
    setFavoriteModalOpen(false)
    window.__pendingFavoriteMatrix = null
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

  return (<>
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
          onFavorite={handleFavorite}
        />
        
        <FeatureGrid />
      </div>
    </main>
    <FavoriteModal open={favoriteModalOpen} defaultName={favoriteDefaultName} onCancel={() => { setFavoriteModalOpen(false); window.__pendingFavoriteMatrix = null }} onSave={(name) => saveFavorite(name)} />
    {savedMessage && (
      <div className="fixed bottom-6 right-6 bg-black text-white px-4 py-2 rounded shadow">{savedMessage}</div>
    )}
  </>)
}
