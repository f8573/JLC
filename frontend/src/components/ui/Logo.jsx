import React from 'react'

export default function Logo({ variant = 'default' }) {
  const sizes = {
    default: 'size-9',
    large: 'size-12'
  }

  return (
    <div className={`${sizes[variant]} bg-primary flex items-center justify-center rounded-md shadow-lg shadow-primary/20`}>
      <span className="text-white text-[20px] font-bold">Λ</span>
    </div>
  )
}
