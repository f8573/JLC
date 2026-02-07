import { useState, useEffect, useCallback } from 'react'

/**
 * Hook to manage favorite matrices stored in localStorage.
 *
 * @param {string} matrixString - The matrix identifier
 * @param {Object} diagnostics - The diagnostics data for the matrix
 * @returns {Object} Favorite state and handlers
 */
export function useFavorites(matrixString, diagnostics) {
  const [isFavorited, setIsFavorited] = useState(false)
  const [favoriteModalOpen, setFavoriteModalOpen] = useState(false)
  const [favoriteDefaultName, setFavoriteDefaultName] = useState('')
  const [savedMessage, setSavedMessage] = useState('')

  useEffect(() => {
    try {
      const raw = localStorage.getItem('favorites')
      const arr = raw ? JSON.parse(raw) : []
      const found = arr.find(f => f.matrixString === matrixString)
      setIsFavorited(!!found)
    } catch {
      setIsFavorited(false)
    }
  }, [matrixString])

  const openFavorite = useCallback(() => {
    setFavoriteDefaultName(matrixString.slice(0, 40))
    window.__pendingFavoriteMatrix = { matrixString, diagnostics }
    setFavoriteModalOpen(true)
  }, [matrixString, diagnostics])

  const saveFavorite = useCallback((name) => {
    try {
      const diag = window.__pendingFavoriteMatrix?.diagnostics || diagnostics
      const r = diag?.rows || diag?.dimensions?.rows || 0
      const c = diag?.columns || diag?.dimensions?.cols || 0
      const density = diag?.density ?? 0
      let type = 'general'
      if (diag?.isIdentity) type = 'identity'
      else if (diag?.symmetric) type = 'symmetric'
      else if (diag?.hessenberg) type = 'hessenberg'
      const key = 'favorites'
      const raw = localStorage.getItem(key)
      const arr = raw ? JSON.parse(raw) : []
      const existing = arr.findIndex(item => item.matrixString === matrixString)
      const entry = { name, matrixString, rows: r, cols: c, density, type, ts: Date.now() }
      if (existing >= 0) {
        arr[existing] = { ...arr[existing], ...entry }
      } else {
        arr.unshift(entry)
      }
      // dedupe by matrixString
      const deduped = []
      for (const it of arr) {
        if (!deduped.find(d => d.matrixString === it.matrixString)) deduped.push(it)
      }
      localStorage.setItem(key, JSON.stringify(deduped))
      setSavedMessage('Saved to favorites')
      setTimeout(() => setSavedMessage(''), 2500)
      setIsFavorited(true)
    } catch {
      // ignore
    }
    setFavoriteModalOpen(false)
    window.__pendingFavoriteMatrix = null
  }, [matrixString, diagnostics])

  const removeFavorite = useCallback(() => {
    try {
      const key = 'favorites'
      const raw = localStorage.getItem(key)
      const arr = raw ? JSON.parse(raw) : []
      const idx = arr.findIndex(f => f.matrixString === matrixString)
      if (idx >= 0) {
        arr.splice(idx, 1)
        localStorage.setItem(key, JSON.stringify(arr))
      }
    } catch {
      // ignore
    }
    setIsFavorited(false)
  }, [matrixString])

  const cancelFavorite = useCallback(() => {
    setFavoriteModalOpen(false)
    window.__pendingFavoriteMatrix = null
  }, [])

  return {
    isFavorited,
    favoriteModalOpen,
    favoriteDefaultName,
    savedMessage,
    openFavorite,
    saveFavorite,
    removeFavorite,
    cancelFavorite,
  }
}
