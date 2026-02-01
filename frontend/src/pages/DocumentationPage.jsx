import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

/**
 * Documentation page - currently under construction.
 */
export default function DocumentationPage() {
  const handleCompute = useMatrixCompute()

  return (
    <div className="bg-background-light dark:bg-slate-900 font-display text-slate-900 dark:text-slate-100 h-screen overflow-hidden transition-colors duration-300">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="documentation" />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light dark:bg-slate-900 transition-colors duration-300">
          <div className="max-w-[800px] mx-auto p-8">
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <div className="size-32 bg-primary/10 dark:bg-primary/20 rounded-full flex items-center justify-center mb-8">
                <span className="material-symbols-outlined text-primary text-[64px]">construction</span>
              </div>
              <h1 className="text-4xl font-black tracking-tight text-slate-900 dark:text-white uppercase mb-4">Under Construction</h1>
              <p className="text-lg text-slate-500 dark:text-slate-400 max-w-md mb-8">
                We're building something amazing! Our documentation is being crafted with care and will be available soon.
              </p>
              <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 p-8 w-full max-w-md transition-colors duration-300">
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-emerald-600 dark:text-emerald-400 text-[20px]">check_circle</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800 dark:text-slate-200">Core Computation Engine</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400">Fully operational</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-blue-600 dark:text-blue-400 text-[20px]">science</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800 dark:text-slate-200">Matrix Analysis</p>
                      <p className="text-xs text-blue-600 dark:text-blue-400">In Preview / Beta</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-amber-100 dark:bg-amber-900/30 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-amber-600 dark:text-amber-400 text-[20px]">pending</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800 dark:text-slate-200">Documentation</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400">Coming soon</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-red-600 dark:text-red-400 text-[20px]">cancel</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800 dark:text-slate-200">API Access</p>
                      <p className="text-xs text-red-600 dark:text-red-400">Not started</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-8 flex gap-4">
                <a 
                  href="/" 
                  className="px-6 py-2.5 bg-primary text-white rounded-xl text-sm font-bold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all active:scale-95"
                >
                  Back to Home
                </a>
                <a 
                  href="/settings" 
                  className="px-6 py-2.5 rounded-xl text-sm font-bold text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                >
                  Go to Settings
                </a>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
