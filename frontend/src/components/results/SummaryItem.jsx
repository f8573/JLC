import React from 'react'

export default function SummaryItem({ title, description }) {
  return (
    <li className="flex items-start gap-3">
      <span className="material-symbols-outlined text-primary text-sm mt-1">check_circle</span>
      <div>
        <p className="text-sm font-bold">{title}</p>
        <p className="text-xs text-slate-500">{description}</p>
      </div>
    </li>
  )
}
