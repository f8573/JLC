import React from 'react'
import { computeAccuracySeverity } from '../../utils/accuracySeverity'

/**
 * Accuracy badge component that displays validation status.
 *
 * @param {Object} props
 * @param {Object} props.validation - Validation object from API
 * @param {boolean} [props.compact=false] - Use compact (emoji only) display
 * @param {Function} [props.onInfoClick] - Click handler for the info button
 * @param {string} [props.className=''] - Additional CSS classes
 */
export default function AccuracyBadge({ validation, compact = false, onInfoClick, className = '' }) {
  const severity = computeAccuracySeverity(validation)

  const bgColors = {
    critical: 'bg-red-50 border-red-200 dark:bg-red-900/30 dark:border-red-700/60',
    severe: 'bg-orange-50 border-orange-200 dark:bg-orange-900/30 dark:border-orange-700/60',
    moderate: 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/30 dark:border-yellow-700/60',
    mild: 'bg-blue-50 border-blue-200 dark:bg-blue-900/30 dark:border-blue-700/60',
    safe: 'bg-emerald-50 border-emerald-200 dark:bg-emerald-900/30 dark:border-emerald-700/60'
  }

  const textColors = {
    critical: 'text-red-700 dark:text-red-300',
    severe: 'text-orange-700 dark:text-orange-300',
    moderate: 'text-yellow-700 dark:text-yellow-300',
    mild: 'text-blue-700 dark:text-blue-300',
    safe: 'text-emerald-700 dark:text-emerald-300'
  }

  const infoButtonColors = {
    critical: 'text-red-500 hover:text-red-700 hover:bg-red-100 dark:text-red-300 dark:hover:text-red-200 dark:hover:bg-red-900/40',
    severe: 'text-orange-500 hover:text-orange-700 hover:bg-orange-100 dark:text-orange-300 dark:hover:text-orange-200 dark:hover:bg-orange-900/40',
    moderate: 'text-yellow-600 hover:text-yellow-800 hover:bg-yellow-100 dark:text-yellow-300 dark:hover:text-yellow-200 dark:hover:bg-yellow-900/40',
    mild: 'text-blue-500 hover:text-blue-700 hover:bg-blue-100 dark:text-blue-300 dark:hover:text-blue-200 dark:hover:bg-blue-900/40',
    safe: 'text-emerald-500 hover:text-emerald-700 hover:bg-emerald-100 dark:text-emerald-300 dark:hover:text-emerald-200 dark:hover:bg-emerald-900/40'
  }

  return (
    <div className={`inline-flex items-center gap-1 ${className}`}>
      <span
        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-xs font-semibold ${bgColors[severity.level]} ${textColors[severity.level]}`}
        title={validation?.message || 'No validation data'}
      >
        {compact ? severity.shortLabel : severity.label}
      </span>
      {onInfoClick && (
        <button
          onClick={onInfoClick}
          className={`inline-flex items-center justify-center w-5 h-5 rounded-full transition-colors ${infoButtonColors[severity.level]}`}
          title="View decomposition details"
        >
          <span className="text-xs font-bold">❓</span>
        </button>
      )}
    </div>
  )
}
