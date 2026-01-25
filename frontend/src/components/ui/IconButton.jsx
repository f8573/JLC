import React from 'react'

export default function IconButton({ icon, onClick, className = '', variant = 'ghost' }) {
  const variants = {
    ghost: 'p-2 text-slate-gray hover:text-primary hover:bg-purple-light rounded-md transition-all',
    small: 'size-7 rounded bg-white hover:bg-purple-light border border-border-color flex items-center justify-center transition-colors text-slate-400 hover:text-primary'
  }

  return (
    <button onClick={onClick} className={`${variants[variant]} ${className}`}>
      <span className="material-symbols-outlined text-lg">{icon}</span>
    </button>
  )
}
