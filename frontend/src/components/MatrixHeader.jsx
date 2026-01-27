import React, { useEffect, useState } from 'react'

/**
 * Matrix header with input box and compute action.
 *
 * @param {Object} props
 * @param {string} [props.inputValue]
 * @param {(matrixString: string) => void} [props.onCompute]
 */
export default function MatrixHeader({ inputValue, onCompute }) {
  const [value, setValue] = useState(inputValue || '')

  useEffect(() => {
    setValue(inputValue || '')
  }, [inputValue])

  function handleSubmit() {
    if (onCompute) {
      onCompute(value)
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-solid border-slate-200 bg-white px-6 py-3 sticky top-0 z-50">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-3">
          <div className="size-9 bg-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-primary/20">
            <span className="text-xl font-bold italic">Λ</span>
          </div>
          <h2 className="text-lg font-bold leading-tight tracking-tight hidden md:block text-slate-800">MatrixSolve AI</h2>
        </div>
        <div className="flex flex-col min-w-[320px] lg:min-w-[600px]">
          <div className="flex w-full items-stretch rounded-xl h-11 border border-slate-200 bg-slate-50 overflow-hidden focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary transition-all">
            <div className="text-slate-400 flex items-center justify-center pl-4">
              <span className="material-symbols-outlined text-[20px]">function</span>
            </div>
            <input
              className="w-full bg-transparent border-none focus:ring-0 text-sm font-medium px-4 placeholder:text-slate-400"
              placeholder="Enter Matrix (e.g. [[1,2],[3,4]])"
              value={value}
              onChange={(e) => setValue(e.target.value)}
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
          <a className="text-sm font-semibold text-primary" href="#">Documentation</a>
        </nav>
        <div className="h-8 w-[1px] bg-slate-200 mx-2"></div>
        <button className="p-2 hover:bg-primary/5 rounded-lg transition-colors text-slate-500">
          <span className="material-symbols-outlined">settings</span>
        </button>
        <div className="size-9 rounded-full bg-primary/10 flex items-center justify-center overflow-hidden border border-primary/20">
          <img
            alt="User Profile Avatar"
            className="w-full h-full object-cover"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuC70Zlk9Ogs5MfHabNJIW4Sm_CwO6ncEB0q4KWOrv6c6ocrvlqXrJHK624PUR5QZZATxIHkjXtNlgsgj_UQQdo-egnAxxN5LbJ4L58yvxlnoliW_08XgkfbRYGw1Wzj-uNHDJ0vYBdzHQyHTcv1SxbUpyldCmGgqcsEahxdxOqXb8vDAwn3nWP146fCJf_ygVT896SppHpo33mGRkewEtu7YaILe_3Smxn9Istj86EXFUzGyzOeCHb9yjC0gQ72hqUg-sKK3x5-xZg_"
          />
        </div>
      </div>
    </header>
  )
}
