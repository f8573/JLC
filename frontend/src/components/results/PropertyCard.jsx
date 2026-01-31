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
    <div className="p-5 rounded-xl bg-primary-light border border-primary/10 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className={`size-10 rounded-lg ${iconBg} flex items-center justify-center ${iconColor}`}>
          <span className="material-symbols-outlined">{icon}</span>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-500 uppercase">{label}</p>
          <div className="text-xl font-bold math-font">{value}</div>
        </div>
      </div>
    </div>
  )
}
