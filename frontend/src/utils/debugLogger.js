/**
 * Simple debug logging helpers for matrix analysis events.
 * Use `logMatrixAnalysis(matrix, stage, info)` with stages: 'start'|'end'|'error'.
 */

function shapeOf(matrix) {
  if (!Array.isArray(matrix)) return 'unknown'
  const rows = matrix.length
  const cols = Array.isArray(matrix[0]) ? matrix[0].length : 0
  return `${rows}x${cols}`
}

function preview(matrix, maxRows = 3, maxCols = 3) {
  if (!Array.isArray(matrix)) return null
  return matrix.slice(0, maxRows).map((r) => (Array.isArray(r) ? r.slice(0, maxCols) : r))
}

export function logMatrixAnalysis(matrix, stage = 'start', info = {}) {
  try {
    // check enabled flag for non-error logs
    function isMatrixLoggingEnabled() {
      try {
        const v = sessionStorage.getItem('matrixLogging')
        if (v === null) return true
        return v === 'true'
      } catch {
        return true
      }
    }

    function setMatrixLogging(enabled) {
      try {
        sessionStorage.setItem('matrixLogging', enabled ? 'true' : 'false')
      } catch {}
    }

    const when = new Date().toISOString()
    const shape = shapeOf(matrix)
    const pv = preview(matrix)
    if (stage === 'error') {
      console.error(`[MatrixAnalysis] ERROR ${when} shape=${shape}`, info || {}, { preview: pv })
      return
    }

    if (!isMatrixLoggingEnabled()) return

    if (stage === 'start') {
      console.log(`[MatrixAnalysis] START ${when} shape=${shape}`)
      console.log('Preview:', pv)
    } else if (stage === 'end') {
      console.log(`[MatrixAnalysis] END   ${when} shape=${shape} durationMs=${info && info.durationMs ? info.durationMs : 'unknown'}`)
      if (info && info.summary) console.log('Summary:', info.summary)
      console.log('Preview:', pv)
    } else {
      console.log(`[MatrixAnalysis] ${stage} ${when} shape=${shape}`, info || {})
      console.log('Preview:', pv)
    }
  } catch (e) {
    // never throw from logger
  }
}

export { logMatrixAnalysis }
export function isMatrixLoggingEnabled() {
  try {
    const v = sessionStorage.getItem('matrixLogging')
    if (v === null) return true
    return v === 'true'
  } catch {
    return true
  }
}

export function setMatrixLogging(enabled) {
  try {
    sessionStorage.setItem('matrixLogging', enabled ? 'true' : 'false')
  } catch {}
}

export default { logMatrixAnalysis, isMatrixLoggingEnabled, setMatrixLogging }
