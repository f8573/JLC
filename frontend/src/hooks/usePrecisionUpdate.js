import { useState, useEffect } from 'react'

/**
 * Hook that triggers a re-render whenever the precision setting changes.
 * Components using formatNumber or formatComplex should use this hook
 * to ensure they re-render with the new precision.
 * 
 * @returns {string} Current precision value
 */
export function usePrecisionUpdate() {
  const [precision, setPrecision] = useState(() => localStorage.getItem('precision') || '6')

  useEffect(() => {
    const handlePrecisionChange = (event) => {
      setPrecision(event.detail?.precision || localStorage.getItem('precision') || '6')
    }

    // Also listen for storage changes from other tabs
    const handleStorageChange = (event) => {
      if (event.key === 'precision') {
        setPrecision(event.newValue || '6')
      }
    }

    window.addEventListener('precision-changed', handlePrecisionChange)
    window.addEventListener('storage', handleStorageChange)

    return () => {
      window.removeEventListener('precision-changed', handlePrecisionChange)
      window.removeEventListener('storage', handleStorageChange)
    }
  }, [])

  return precision
}
