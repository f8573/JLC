import React from 'react'

export default function NumberInput({ 
  value, 
  onChange, 
  min = 1, 
  max = 20, 
  label,
  className = '' 
}) {
  return (
    <input
      aria-label={label}
      value={value}
      onChange={onChange}
      className={`w-16 h-10 bg-slate-50 border border-border-color rounded-lg text-center text-sm font-mono text-slate-900 focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all shadow-sm ${className}`}
      type="number"
      min={min}
      max={max}
    />
  )
}
