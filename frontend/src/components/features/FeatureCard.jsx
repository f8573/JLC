import React from 'react'

export default function FeatureCard({ icon, title, description, hasBorder = false }) {
  return (
    <div className={`flex items-start gap-3 p-4 ${hasBorder ? 'border-l border-border-color' : ''}`}>
      <span className="material-symbols-outlined text-primary">{icon}</span>
      <div>
        <h4 className="text-sm font-bold text-slate-900 mb-1">{title}</h4>
        <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}
