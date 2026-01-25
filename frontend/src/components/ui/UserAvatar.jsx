import React from 'react'

export default function UserAvatar({ src, alt = 'User Avatar', size = 'md' }) {
  const sizes = {
    sm: 'size-8',
    md: 'size-9'
  }

  return (
    <div
      className={`bg-slate-100 aspect-square rounded-full ${sizes[size]} border border-border-color bg-cover bg-center`}
      style={{ backgroundImage: `url("${src}")` }}
      role="img"
      aria-label={alt}
    ></div>
  )
}
