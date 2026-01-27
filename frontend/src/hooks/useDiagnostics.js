import { useEffect, useState } from 'react'
import { parseMatrixString, loadCachedDiagnostics, analyzeAndCache } from '../utils/diagnostics'

/**
 * Fetch and cache diagnostics for a given matrix string.
 *
 * @param {string} matrixString
 * @returns {{diagnostics: any, loading: boolean, error: string | null}}
 */
export function useDiagnostics(matrixString) {
  const [diagnostics, setDiagnostics] = useState(() => loadCachedDiagnostics(matrixString))
  const [loading, setLoading] = useState(!diagnostics)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!matrixString) return
    const cached = loadCachedDiagnostics(matrixString)
    if (cached) {
      // Only use cached results immediately if they include the newer per-eigenvalue
      // aggregated information; otherwise force a fresh fetch to avoid showing
      // stale/old schema data.
      if (cached.eigenInformationPerValue) {
        setDiagnostics(cached)
        setLoading(false)
      }
      // continue to fetch fresh diagnostics below to avoid stale cache
    }

    const matrixData = parseMatrixString(matrixString)
    if (!matrixData) {
      setError('Invalid matrix data')
      setLoading(false)
      return
    }

    let active = true
    setLoading(true)
    analyzeAndCache(matrixData, matrixString)
      .then((data) => {
        if (!active) return
        setDiagnostics(data)
        setLoading(false)
      })
      .catch((err) => {
        if (!active) return
        setError(err?.message || 'Diagnostics failed')
        setLoading(false)
      })

    return () => {
      active = false
    }
  }, [matrixString])

  return { diagnostics, loading, error }
}

