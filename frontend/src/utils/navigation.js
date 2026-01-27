/**
 * Push a new history entry and notify listeners.
 *
 * @param {string} path
 */
export function navigate(path) {
  if (window.location.pathname === path) return
  window.history.pushState({}, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

/**
 * Replace the current history entry and notify listeners.
 *
 * @param {string} path
 */
export function replace(path) {
  window.history.replaceState({}, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}
