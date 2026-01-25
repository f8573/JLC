import React from 'react'

export default function Badge({ children, variant = 'default', animated = false }) {
  const variants = {
    default: 'bg-purple-light border border-primary/20 text-primary',
    status: 'bg-primary/10 text-primary'
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${variants[variant]}`}>
      {animated && <span className="flex size-2 rounded-full bg-primary animate-pulse"></span>}
      <span className="text-[11px] font-bold uppercase tracking-wider">{children}</span>
    </div>
  )
}
