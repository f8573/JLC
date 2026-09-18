import React from 'react'

/**
 * Summary list item for key diagnostics.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.title
 * @param {React.ReactNode} props.description
 */
export default function SummaryItem({ title, description }) {
  return (
    <li className="flex items-start gap-3">
      <span className="material-symbols-outlined text-primary text-sm mt-1">check_circle</span>
      <div>
        <p className="text-sm font-bold text-slate-900 dark:text-slate-100">{title}</p>
        <p className="text-xs text-slate-500 dark:text-slate-400">{description}</p>
      </div>
    </li>
  )
}
