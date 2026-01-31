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
    critical: 'bg-red-50 border-red-200',
    severe: 'bg-orange-50 border-orange-200',
    moderate: 'bg-yellow-50 border-yellow-200',
    mild: 'bg-blue-50 border-blue-200',
    safe: 'bg-green-50 border-green-200'
  }

  const textColors = {
    critical: 'text-red-700',
    severe: 'text-orange-700',
    moderate: 'text-yellow-700',
    mild: 'text-blue-700',
    safe: 'text-green-700'
  }

  const infoButtonColors = {
    critical: 'text-red-500 hover:text-red-700 hover:bg-red-100',
    severe: 'text-orange-500 hover:text-orange-700 hover:bg-orange-100',
    moderate: 'text-yellow-600 hover:text-yellow-800 hover:bg-yellow-100',
    mild: 'text-blue-500 hover:text-blue-700 hover:bg-blue-100',
    safe: 'text-green-500 hover:text-green-700 hover:bg-green-100'
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

/**
 * Compact accuracy indicator (just the emoji).
 *
 * @param {Object} props
 * @param {Object} props.validation - Validation object from API
 * @param {string} [props.className=''] - Additional CSS classes
 */
export function AccuracyIndicator({ validation, className = '' }) {
  const severity = computeAccuracySeverity(validation)
  
  return (
    <span
      className={`cursor-help ${className}`}
      title={validation?.message || 'Validation status'}
      style={{ color: severity.color }}
    >
      {severity.shortLabel}
    </span>
  )
}
