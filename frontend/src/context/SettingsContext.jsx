import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'

const SettingsContext = createContext()

const DEFAULT_PRECISION = '6'
const DEFAULT_OUTPUT_FORMAT = 'numeric'

/**
 * Settings provider that manages user preferences like decimal precision.
 * Persists settings to localStorage and broadcasts changes across the app.
 */
export function SettingsProvider({ children }) {
  const [precision, setPrecisionState] = useState(() => {
    return localStorage.getItem('precision') || DEFAULT_PRECISION
  })

  const [outputFormat, setOutputFormatState] = useState(() => {
    return localStorage.getItem('outputFormat') || DEFAULT_OUTPUT_FORMAT
  })

  // Get numeric precision value
  const getPrecisionDigits = useCallback(() => {
    const num = parseInt(precision, 10)
    return isNaN(num) ? 6 : num
  }, [precision])

  const setPrecision = useCallback((newPrecision) => {
    setPrecisionState(newPrecision)
    localStorage.setItem('precision', newPrecision)
    // Dispatch custom event to notify components of precision change
    window.dispatchEvent(new CustomEvent('precision-changed', { detail: { precision: newPrecision } }))
  }, [])

  const setOutputFormat = useCallback((newFormat) => {
    setOutputFormatState(newFormat)
    localStorage.setItem('outputFormat', newFormat)
  }, [])

  const resetDefaults = useCallback(() => {
    setPrecision(DEFAULT_PRECISION)
    setOutputFormat(DEFAULT_OUTPUT_FORMAT)
  }, [setPrecision, setOutputFormat])

  // Clear all search history (recent sessions)
  const clearSearchHistory = useCallback(() => {
    localStorage.removeItem('recentSessions')
    // Dispatch event to notify components that history was cleared
    window.dispatchEvent(new CustomEvent('history-cleared'))
  }, [])

  // Export all matrices from history as JSON
  const exportAllMatrices = useCallback(() => {
    try {
      const raw = localStorage.getItem('recentSessions')
      const sessions = raw ? JSON.parse(raw) : []
      
      // Extract unique matrices by their title (matrix string)
      const seen = new Set()
      const uniqueMatrices = []

      for (const session of sessions) {
        const matrixString = session.title
        if (matrixString && !seen.has(matrixString)) {
          seen.add(matrixString)
          uniqueMatrices.push({
            matrixString: matrixString,
            rows: session.rows || null,
            cols: session.cols || null,
            timestamp: session.ts || null
          })
        }
      }

      // Build export object
      const exportData = {
        exportedAt: new Date().toISOString(),
        totalMatrices: uniqueMatrices.length,
        matrices: uniqueMatrices
      }

      // Create and download JSON file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `matrices_export_${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      return { success: true, count: uniqueMatrices.length }
    } catch (e) {
      console.error('Failed to export matrices:', e)
      return { success: false, error: e.message }
    }
  }, [])

  return (
    <SettingsContext.Provider value={{
      precision,
      setPrecision,
      outputFormat,
      setOutputFormat,
      resetDefaults,
      getPrecisionDigits,
      clearSearchHistory,
      exportAllMatrices
    }}>
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings() {
  const context = useContext(SettingsContext)
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider')
  }
  return context
}
