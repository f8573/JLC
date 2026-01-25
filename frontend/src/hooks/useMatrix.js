import { useState, useCallback } from 'react'

function makeEmpty(rows, cols, fill = '0') {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => fill))
}

export function useMatrix(initialRows = 2, initialCols = 2, initialValues = null) {
  const [rows, setRows] = useState(initialRows)
  const [cols, setCols] = useState(initialCols)
  const [values, setValues] = useState(() => 
    initialValues || [['1', '0'], ['0', '1']]
  )

  const updateDimensions = useCallback((newRows, newCols) => {
    const r = Math.max(1, Math.min(20, Number(newRows) || 1))
    const c = Math.max(1, Math.min(20, Number(newCols) || 1))
    setRows(r)
    setCols(c)
    setValues((prev) => {
      const next = makeEmpty(r, c, '0')
      for (let i = 0; i < Math.min(prev.length, r); i++) {
        for (let j = 0; j < Math.min(prev[0]?.length || 0, c); j++) {
          next[i][j] = prev[i][j]
        }
      }
      return next
    })
  }, [])

  const updateCell = useCallback((rIdx, cIdx, value) => {
    setValues((prev) => {
      const next = prev.map((r) => r.slice())
      next[rIdx][cIdx] = value
      return next
    })
  }, [])

  const transpose = useCallback(() => {
    const transposed = makeEmpty(cols, rows, '0')
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposed[j][i] = values[i]?.[j] ?? '0'
      }
    }
    setValues(transposed)
    setRows(cols)
    setCols(rows)
  }, [rows, cols, values])

  const toMatrixString = useCallback(() => {
    return '[' + values.map((r) => '[' + r.join(', ') + ']').join(', ') + ']'
  }, [values])

  return {
    rows,
    cols,
    values,
    updateDimensions,
    updateCell,
    transpose,
    toMatrixString,
    setValues
  }
}
