import { analyzeAndCache, parseMatrixString, matrixToString } from '../utils/diagnostics'

/**
 * Hook that returns a matrix compute handler for header inputs.
 */
export function useMatrixCompute() {
  return async function handleCompute(inputValue) {
    const parsed = parseMatrixString(inputValue)
    if (!parsed) {
      return
    }
    const normalized = matrixToString(parsed)
    try {
      await analyzeAndCache(parsed, normalized)
    } catch (err) {
      // allow navigation on failure
    }
    window.location.href = `/matrix=${encodeURIComponent(normalized)}/basic`
  }
}

