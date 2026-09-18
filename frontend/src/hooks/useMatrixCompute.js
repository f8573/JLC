import { analyzeAndCache, parseMatrixString, matrixToString } from '../utils/diagnostics'
import { parseMatrixInputText } from '../utils/matrixInput'

/**
 * Hook that returns a matrix compute handler for header inputs.
 */
export function useMatrixCompute() {
  return async function handleCompute(inputValue) {
    let parsed = parseMatrixString(inputValue)
    if (!parsed) {
      try {
        parsed = parseMatrixInputText(inputValue)
      } catch {
        parsed = null
      }
    }

    if (!parsed) {
      return
    }

    const normalized = matrixToString(parsed)
    try {
      await analyzeAndCache(parsed, normalized)
    } catch {
      // allow navigation on failure
    }
    window.location.href = `/matrix=${encodeURIComponent(normalized)}/basic`
  }
}
