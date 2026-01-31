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
    <div className="bg-background-light font-display text-slate-900 h-screen overflow-hidden">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="documentation" />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light">
          <div className="max-w-[800px] mx-auto p-8">
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <div className="size-32 bg-primary/10 rounded-full flex items-center justify-center mb-8">
                <span className="material-symbols-outlined text-primary text-[64px]">construction</span>
              </div>
              <h1 className="text-4xl font-black tracking-tight text-slate-900 uppercase mb-4">Under Construction</h1>
              <p className="text-lg text-slate-500 max-w-md mb-8">
                We're building something amazing! Our documentation is being crafted with care and will be available soon.
              </p>
              <div className="bg-white rounded-2xl border border-slate-200 p-8 w-full max-w-md">
                <div className="space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-emerald-100 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-emerald-600 text-[20px]">check_circle</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800">Core Computation Engine</p>
                      <p className="text-xs text-slate-500">Fully operational</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-blue-600 text-[20px]">science</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800">Matrix Analysis</p>
                      <p className="text-xs text-blue-600">In Preview / Beta</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-amber-100 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-amber-600 text-[20px]">pending</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800">Documentation</p>
                      <p className="text-xs text-slate-500">Coming soon</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="size-10 bg-red-100 rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-red-600 text-[20px]">cancel</span>
                    </div>
                    <div className="text-left">
                      <p className="font-bold text-slate-800">API Access</p>
                      <p className="text-xs text-red-600">Not started</p>
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
                  className="px-6 py-2.5 rounded-xl text-sm font-bold text-slate-600 border border-slate-200 hover:bg-slate-50 transition-colors"
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
