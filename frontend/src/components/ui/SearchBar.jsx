import React from 'react'

/**
 * Search input for command-like queries.
 *
 * @param {Object} props
 * @param {string} [props.placeholder='Enter matrix commands...']
 * @param {string} [props.defaultValue='']
 */
export default function SearchBar({ placeholder = 'Enter matrix commands...', defaultValue = '' }) {
  return (
    <div className="relative group">
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-gray">
        <span className="material-symbols-outlined text-[20px]">search</span>
      </div>
      <input
        className="block w-full bg-slate-50 border border-border-color rounded-lg py-1.5 pl-10 pr-3 text-sm text-slate-900 placeholder-slate-gray focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all"
        placeholder={placeholder}
        type="text"
        defaultValue={defaultValue}
      />
    </div>
  )
}
