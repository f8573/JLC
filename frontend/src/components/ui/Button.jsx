import React from 'react'

/**
 * Primary button component with optional icon.
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children
 * @param {(event: React.MouseEvent<HTMLButtonElement>) => void} [props.onClick]
 * @param {'primary'|'secondary'} [props.variant='primary']
 * @param {string} [props.icon]
 * @param {string} [props.className='']
 */
export default function Button({ 
  children, 
  onClick, 
  variant = 'primary', 
  icon,
  className = '',
  ...props 
}) {
  const variants = {
    primary: 'bg-primary text-white hover:bg-primary-hover shadow-lg shadow-primary/20',
    secondary: 'bg-white dark:bg-slate-700 border border-border-color dark:border-slate-600 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-600',
    ghost: 'bg-transparent text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
  }

  return (
    <button 
      onClick={onClick} 
      className={`flex items-center justify-center gap-3 px-8 py-4 text-base font-bold rounded-lg transition-all group ${variants[variant] || variants.primary} ${className}`}
      {...props}
    >
      {icon && (
        <span className="material-symbols-outlined text-xl group-hover:scale-110 transition-transform">
          {icon}
        </span>
      )}
      {children}
    </button>
  )
}
