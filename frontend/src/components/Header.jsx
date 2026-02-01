import React, { useState, useEffect } from 'react'

/**
 * Unified top navigation header across all pages.
 * Accepts an optional inputValue for matrix input and onCompute callback.
 *
 * @param {Object} props
 * @param {string} [props.inputValue]
 * @param {(matrixString: string) => void} [props.onCompute]
 */
export default function Header({ inputValue: initialValue, onCompute }) {
  const [inputValue, setInputValue] = useState(initialValue || '')

  useEffect(() => {
    setInputValue(initialValue || '')
  }, [initialValue])

  function handleSubmit() {
    if (inputValue.trim()) {
      if (onCompute) {
        onCompute(inputValue.trim())
      } else {
        window.location.href = `/matrix=${encodeURIComponent(inputValue.trim())}/basic`
      }
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-solid border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-3 sticky top-0 z-50 transition-colors duration-300">
      <div className="flex items-center gap-8">
        <a href="/" className="flex items-center gap-3">
          <div className="size-9 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/20">
            <span className="text-xl font-bold italic">Λ</span>
          </div>
          <h2 className="text-lg font-bold leading-tight tracking-tight hidden md:block text-slate-800 dark:text-white">ΛCompute</h2>
        </a>
        <div className="flex flex-col min-w-[320px] lg:min-w-[600px]">
          <div className="flex w-full items-stretch rounded-xl h-11 border border-slate-200 dark:border-slate-600 bg-slate-50 dark:bg-slate-700 overflow-hidden focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all">
            <div className="text-slate-400 dark:text-slate-500 flex items-center justify-center pl-4">
              <span className="material-symbols-outlined text-[20px]">function</span>
            </div>
            <input
              className="w-full bg-transparent border-none focus:ring-0 text-sm font-medium px-4 placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-900 dark:text-slate-100"
              placeholder="Enter Matrix (e.g. [[1,2],[3,4]])"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
            />
            <button
              className="bg-primary text-white px-8 text-xs font-bold uppercase tracking-widest hover:bg-primary/90 transition-all active:scale-95"
              onClick={handleSubmit}
            >
              Compute
            </button>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <nav className="hidden xl:flex items-center gap-6">
          <a className="text-sm font-semibold text-slate-600 dark:text-slate-300 hover:text-primary transition-colors" href="/documentation">Documentation</a>
        </nav>
        <div className="h-8 w-[1px] bg-slate-200 dark:bg-slate-600 mx-2"></div>
        <a href="/settings" className="p-2 hover:bg-primary/5 dark:hover:bg-primary/20 rounded-lg transition-colors text-slate-500 dark:text-slate-400">
          <span className="material-symbols-outlined">settings</span>
        </a>
      </div>
    </header>
  )
}
