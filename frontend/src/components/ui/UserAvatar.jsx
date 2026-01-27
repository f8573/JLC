import React from 'react'

/**
 * User avatar badge rendered as a background image.
 *
 * @param {Object} props
 * @param {string} props.src
 * @param {string} [props.alt='User Avatar']
 * @param {'sm'|'md'} [props.size='md']
 */
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
