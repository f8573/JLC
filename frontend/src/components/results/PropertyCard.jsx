import React from 'react'

/**
 * Summary property card for matrix metrics.
 *
 * @param {Object} props
 * @param {string} props.icon
 * @param {string} props.label
 * @param {React.ReactNode} props.value
 * @param {string} [props.iconBg='bg-primary/10']
 * @param {string} [props.iconColor='text-primary']
 */
export default function PropertyCard({ icon, label, value, iconBg = 'bg-primary/10', iconColor = 'text-primary' }) {
  return (
    <div className="p-5 rounded-xl bg-primary-light dark:bg-primary/10 border border-primary/10 dark:border-primary/20 flex items-center justify-between transition-colors duration-300">
      <div className="flex items-center gap-4">
        <div className={`size-10 rounded-lg ${iconBg} flex items-center justify-center ${iconColor}`}>
          <span className="material-symbols-outlined">{icon}</span>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase">{label}</p>
          <div className="text-xl font-bold math-font text-slate-900 dark:text-slate-100">{value}</div>
        </div>
      </div>
    </div>
  )
}
