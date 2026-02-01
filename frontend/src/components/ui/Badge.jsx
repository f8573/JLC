import React from 'react'

/**
 * Status badge component.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children
 * @param {'default'|'status'} [props.variant='default']
 * @param {boolean} [props.animated=false]
 */
export default function Badge({ children, variant = 'default', animated = false }) {
  const variants = {
    default: 'bg-purple-light dark:bg-primary/20 border border-primary/20 text-primary',
    status: 'bg-primary/10 dark:bg-primary/20 text-primary'
  }

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full transition-colors duration-300 ${variants[variant]}`}>
      {animated && <span className="flex size-2 rounded-full bg-primary animate-pulse"></span>}
      <span className="text-[11px] font-bold uppercase tracking-wider">{children}</span>
    </div>
  )
}
