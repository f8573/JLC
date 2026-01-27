import { useRef, useCallback } from 'react'

/**
 * Hook that owns the input grid ref and a transpose animation helper.
 */
export function useMatrixAnimation() {
  const containerRef = useRef(null)

  const animateTranspose = useCallback((rows, cols, onDone) => {
    const container = containerRef.current
    if (!container) return onDone()
    const cells = Array.from(container.querySelectorAll('[data-cell]'))
    if (cells.length === 0) return onDone()

    // Animation removed: just ensure originals visible and proceed
    cells.forEach((el) => {
      el.style.visibility = ''
    })

    requestAnimationFrame(() => onDone())
  }, [])

  return { containerRef, animateTranspose }
}
