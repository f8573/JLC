import React from 'react'

/**
 * Focus a specific matrix cell input by row/column.
 *
 * @param {number} r
 * @param {number} c
 * @returns {HTMLInputElement | null}
 */
function focusCell(r, c) {
  const el = document.querySelector(`[data-cell][data-row="${r}"][data-col="${c}"]`)
  if (el) {
    el.focus()
    return el
  }
  return null
}

/**
 * Single matrix input cell with keyboard navigation and validation.
 *
 * @param {Object} props
 * @param {string|number} props.value
 * @param {(row: number, col: number, value: string) => void} props.onChange
 * @param {number} props.rowIndex
 * @param {number} props.colIndex
 * @param {number} props.rows
 * @param {number} props.cols
 * @param {() => void} [props.onAnalyze]
 */
export default function MatrixCell({ value, onChange, rowIndex, colIndex, rows, cols, onAnalyze }) {
  const allowed = /[0-9.\-]/

  /**
   * Normalize raw input by restricting to numeric characters and one leading minus sign.
   *
   * @param {string} raw
   * @returns {string}
   */
  function sanitizeValue(raw) {
    let val = (raw || '').split('').filter((ch) => allowed.test(ch)).join('')
    if (val.startsWith('-')) {
      val = '-' + val.slice(1).replace(/-/g, '')
    } else {
      val = val.replace(/-/g, '')
    }
    return val
  }

  /**
   * Handle user input edits and forward sanitized values.
   *
   * @param {React.ChangeEvent<HTMLInputElement>} e
   */
  function handleChange(e) {
    const cleaned = sanitizeValue(e.target.value)
    if (cleaned !== e.target.value) {
      e.target.value = cleaned
    }
    onChange(rowIndex, colIndex, cleaned)
  }

  /**
   * Keyboard navigation and entry restrictions for the matrix grid.
   *
   * @param {React.KeyboardEvent<HTMLInputElement>} e
   */
  function handleKeyDown(e) {
    const key = e.key
    const el = e.target
    const selStart = el.selectionStart ?? 0
    const selEnd = el.selectionEnd ?? 0
    const len = (el.value || '').length

    if (key === 'Enter') {
      e.preventDefault()
      if (onAnalyze) onAnalyze()
      return
    }

    if (key === '-') {
      // only allow inserting '-' at leading position if not already present
      if (!(selStart === 0 && selEnd === 0 && !(el.value || '').startsWith('-'))) {
        e.preventDefault()
      }
      return
    }

    if (key === 'ArrowLeft') {
      if (selStart === 0 && selEnd === 0) {
        e.preventDefault()
        if (colIndex > 0) {
          const target = focusCell(rowIndex, colIndex - 1)
          if (target) target.setSelectionRange((target.value||'').length, (target.value||'').length)
        } else if (rowIndex > 0) {
          const target = focusCell(rowIndex - 1, cols - 1)
          if (target) target.setSelectionRange((target.value||'').length, (target.value||'').length)
        }
      }
      return
    }

    if (key === 'ArrowRight') {
      if (selStart === len && selEnd === len) {
        e.preventDefault()
        if (colIndex < cols - 1) {
          const target = focusCell(rowIndex, colIndex + 1)
          if (target) target.setSelectionRange(0, 0)
        } else if (rowIndex < rows - 1) {
          const target = focusCell(rowIndex + 1, 0)
          if (target) target.setSelectionRange(0, 0)
        }
      }
      return
    }

    if (key === 'ArrowUp') {
      e.preventDefault()
      if (rowIndex > 0) {
        const target = focusCell(rowIndex - 1, Math.min(colIndex, cols - 1))
        if (target) {
          const pos = Math.min(selStart, (target.value||'').length)
          target.setSelectionRange(pos, pos)
        }
      }
      return
    }

    if (key === 'ArrowDown') {
      e.preventDefault()
      if (rowIndex < rows - 1) {
        const target = focusCell(rowIndex + 1, Math.min(colIndex, cols - 1))
        if (target) {
          const pos = Math.min(selStart, (target.value||'').length)
          target.setSelectionRange(pos, pos)
        }
      }
      return
    }

    // filter other printable characters
    if (key.length === 1 && !allowed.test(key) && !e.ctrlKey && !e.metaKey) {
      e.preventDefault()
      return
    }
  }

  /**
   * Paste handler that sanitizes pasted data.
   *
   * @param {React.ClipboardEvent<HTMLInputElement>} e
   */
  function handlePaste(e) {
    e.preventDefault()
    const text = e.clipboardData.getData('text') || ''
    const newValRaw = (value || '').slice(0, (e.target.selectionStart||0)) + text + (value || '').slice((e.target.selectionEnd||0))
    const newVal = sanitizeValue(newValRaw)
    onChange(rowIndex, colIndex, newVal)
  }

  return (
    <input
      data-cell
      data-row={rowIndex}
      data-col={colIndex}
      className="w-20 h-20 bg-white dark:bg-slate-700 border border-border-color dark:border-slate-600 rounded-lg text-center text-2xl font-mono text-slate-900 dark:text-slate-100 focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all shadow-sm"
      type="text"
      inputMode="decimal"
      value={value}
      onChange={handleChange}
      onKeyDown={handleKeyDown}
      onPaste={handlePaste}
    />
  )
}
