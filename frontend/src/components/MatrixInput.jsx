import React from 'react'
import FavoriteModal from './ui/FavoriteModal'
import { useMatrix } from '../hooks/useMatrix'
import { useMatrixAnimation } from '../hooks/useMatrixAnimation'
import { analyzeAndCache, valuesToMatrixData, matrixToString } from '../utils/diagnostics'
import { parseMatrixInputText } from '../utils/matrixInput'
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
  const [uploadMessage, setUploadMessage] = React.useState('')
  const [uploadError, setUploadError] = React.useState('')
  const [uploadBusy, setUploadBusy] = React.useState(false)
  const fileInputRef = React.useRef(null)

  function storeRecentSession(matrixString) {
    try {
      const key = 'recentSessions'
      const raw = localStorage.getItem(key)
      const arr = raw ? JSON.parse(raw) : []
      arr.unshift({ title: matrixString, ts: Date.now() })
      localStorage.setItem(key, JSON.stringify(arr.slice(0, 10)))
    } catch (e) {
      // ignore
    }
  }

  async function navigateToAnalysis(matrixData, matrixString, withBasicTab = true) {
    storeRecentSession(matrixString)
    try {
      await analyzeAndCache(matrixData, matrixString)
    } catch (e) {
      // allow navigation even if diagnostics failed
    }
    const target = '/matrix=' + encodeURIComponent(matrixString) + (withBasicTab ? '/basic' : '')
    window.location.href = target
  }

  async function handleAnalyze() {
    const matrixData = valuesToMatrixData(values)
    const matrixString = matrixToString(matrixData)
    await navigateToAnalysis(matrixData, matrixString, true)
  }

  function handleUploadClick() {
    setUploadError('')
    setUploadMessage('')
    fileInputRef.current?.click()
  }

  async function handleUploadFileChange(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    setUploadBusy(true)
    setUploadError('')
    setUploadMessage('')

    try {
      const rawContent = await file.text()
      const parsedMatrix = parseMatrixInputText(rawContent)
      const matrixString = matrixToString(parsedMatrix)
      setUploadMessage(`Loaded ${file.name}`)
      await navigateToAnalysis(parsedMatrix, matrixString, false)
    } catch (err) {
      setUploadError(err?.message || 'Failed to parse uploaded matrix file')
    } finally {
      setUploadBusy(false)
    }
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
    <main className="flex-1 overflow-y-auto bg-white dark:bg-slate-900 math-grid relative flex flex-col items-center transition-colors duration-300">
      <div className="w-full max-w-4xl px-8 pt-16 pb-24 flex flex-col items-center">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-extrabold text-slate-900 dark:text-white tracking-tight mb-4 mt-6">
            Matrix Input Console
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg max-w-lg mx-auto">
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
          onUpload={handleUploadClick}
          uploadBusy={uploadBusy}
        />

        {(uploadMessage || uploadError) && (
          <div className="w-full max-w-md mt-4 text-center">
            {uploadMessage && <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400">{uploadMessage}</p>}
            {uploadError && <p className="text-sm font-medium text-rose-600 dark:text-rose-400">{uploadError}</p>}
          </div>
        )}
        
        <FeatureGrid />
      </div>
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleUploadFileChange}
      />
    </main>
    <FavoriteModal open={favoriteModalOpen} defaultName={favoriteDefaultName} onCancel={() => { setFavoriteModalOpen(false); window.__pendingFavoriteMatrix = null }} onSave={(name) => saveFavorite(name)} />
    {savedMessage && (
      <div className="fixed bottom-6 right-6 bg-black dark:bg-slate-700 text-white px-4 py-2 rounded shadow">{savedMessage}</div>
    )}
  </>)
}
