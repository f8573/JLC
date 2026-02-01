import React from 'react'

/**
 * Icon-only button used across the UI.
 *
 * @param {Object} props
 * @param {string} props.icon
 * @param {(event: React.MouseEvent<HTMLButtonElement>) => void} [props.onClick]
 * @param {string} [props.className='']
 * @param {'ghost'|'small'} [props.variant='ghost']
 */
export default function IconButton({ icon, onClick, className = '', variant = 'ghost' }) {
  const variants = {
    ghost: 'p-2 text-slate-gray dark:text-slate-400 hover:text-primary hover:bg-purple-light dark:hover:bg-primary/20 rounded-md transition-all',
    small: 'size-7 rounded bg-white dark:bg-slate-700 hover:bg-purple-light dark:hover:bg-primary/20 border border-border-color dark:border-slate-600 flex items-center justify-center transition-colors text-slate-400 dark:text-slate-500 hover:text-primary'
  }

  return (
    <button onClick={onClick} className={`${variants[variant]} ${className}`}>
      <span className="material-symbols-outlined text-lg">{icon}</span>
    </button>
  )
}
