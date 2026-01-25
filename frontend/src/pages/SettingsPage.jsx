import React from 'react'
import MatrixHeader from '../components/MatrixHeader'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

export default function SettingsPage() {
  const handleCompute = useMatrixCompute()
  return (
    <div className="bg-background-light font-display text-slate-900 min-h-screen">
      <MatrixHeader inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <aside className="w-64 border-r border-slate-200 hidden lg:flex flex-col bg-white p-4 overflow-y-auto">
          <div className="flex flex-col h-full justify-between">
            <div className="space-y-8">
              <div>
                <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4 px-3">Library</h3>
                <div className="space-y-1">
                  <a className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-primary/5 text-slate-600 transition-colors group" href="#">
                    <span className="material-symbols-outlined text-[20px]">analytics</span>
                    <span className="text-sm font-medium">Current Analysis</span>
                  </a>
                  <a className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-primary/5 text-slate-600 transition-colors group" href="/recent">
                    <span className="material-symbols-outlined text-[20px]">history</span>
                    <span className="text-sm font-medium">Recent Queries</span>
                  </a>
                  <a className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-primary/5 text-slate-600 transition-colors group" href="/favorites">
                    <span className="material-symbols-outlined text-[20px]">star</span>
                    <span className="text-sm font-medium">Favorites</span>
                  </a>
                </div>
              </div>
              <div>
                <h3 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-4 px-3">Example Matrices</h3>
                <div className="space-y-1">
                  <button className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-primary/5 text-sm transition-colors text-slate-600">
                    <span>Identity Matrix</span>
                    <span className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded font-mono">I_n</span>
                  </button>
                  <button className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-primary/5 text-sm transition-colors text-slate-600">
                    <span>Hilbert Matrix</span>
                    <span className="material-symbols-outlined text-[16px]">chevron_right</span>
                  </button>
                </div>
              </div>
            </div>
            <div className="pt-4 border-t border-slate-100">
              <a className="flex items-center gap-3 px-3 py-2 rounded-lg bg-primary/10 text-primary group" href="/settings">
                <span className="material-symbols-outlined text-[20px]">settings</span>
                <span className="text-sm font-semibold">Settings</span>
              </a>
            </div>
          </div>
        </aside>
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light">
          <div className="max-w-[800px] mx-auto p-8 space-y-10">
            <div className="space-y-1">
              <h1 className="text-3xl font-black tracking-tight text-slate-900 uppercase">Settings</h1>
              <p className="text-slate-500 text-sm">Configure your personal preferences and computational environment.</p>
            </div>
            <section className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[18px]">palette</span>
                  Appearance
                </h2>
              </div>
              <div className="p-6 space-y-8">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-bold text-slate-800">Theme Mode</p>
                    <p className="text-xs text-slate-500">Switch between light and dark visual interfaces</p>
                  </div>
                  <div className="flex items-center bg-slate-100 p-1 rounded-xl">
                    <button className="px-4 py-1.5 text-xs font-bold rounded-lg bg-white shadow-sm text-primary">Light</button>
                    <button className="px-4 py-1.5 text-xs font-bold rounded-lg text-slate-400 hover:text-slate-600">Dark</button>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="font-bold text-slate-800">Accent Color</p>
                    <p className="text-xs text-slate-500">Choose your primary theme color (shades of purple)</p>
                  </div>
                  <div className="flex gap-4">
                    <button className="size-10 rounded-full bg-[#7c3aed] ring-4 ring-primary/20 border-2 border-white"></button>
                    <button className="size-10 rounded-full bg-[#9333ea] hover:scale-110 transition-transform"></button>
                    <button className="size-10 rounded-full bg-[#a855f7] hover:scale-110 transition-transform"></button>
                    <button className="size-10 rounded-full bg-[#c084fc] hover:scale-110 transition-transform"></button>
                    <button className="size-10 rounded-full bg-[#6366f1] hover:scale-110 transition-transform"></button>
                  </div>
                </div>
              </div>
            </section>
            <section className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[18px]">calculate</span>
                  Computational Preferences
                </h2>
              </div>
              <div className="p-6 space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="space-y-3">
                    <label className="font-bold text-slate-800 block">Decimal Precision</label>
                    <select
                      className="w-full bg-slate-50 border-slate-200 rounded-xl text-sm focus:ring-primary focus:border-primary"
                      defaultValue="6"
                    >
                      <option value="2">2 decimal places</option>
                      <option value="4">4 decimal places</option>
                      <option value="6">6 decimal places</option>
                      <option value="sci">Scientific notation</option>
                    </select>
                    <p className="text-[11px] text-slate-400 italic">Affects numeric displays only.</p>
                  </div>
                  <div className="space-y-3">
                    <label className="font-bold text-slate-800 block">Output Format</label>
                    <div className="flex gap-2">
                      <label className="flex-1 cursor-pointer">
                        <input defaultChecked className="hidden peer" name="format" type="radio" />
                        <div className="p-3 text-center rounded-xl border border-slate-200 peer-checked:border-primary peer-checked:bg-primary/5 peer-checked:text-primary transition-all">
                          <span className="text-xs font-bold">Symbolic</span>
                        </div>
                      </label>
                      <label className="flex-1 cursor-pointer">
                        <input className="hidden peer" name="format" type="radio" />
                        <div className="p-3 text-center rounded-xl border border-slate-200 peer-checked:border-primary peer-checked:bg-primary/5 peer-checked:text-primary transition-all">
                          <span className="text-xs font-bold">Numeric</span>
                        </div>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </section>
            <section className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50">
                <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[18px]">person</span>
                  Account
                </h2>
              </div>
              <div className="p-6 space-y-8">
                <div className="flex items-center gap-6">
                  <div className="relative group">
                    <img
                      alt="User Profile"
                      className="size-20 rounded-full border-2 border-slate-100 object-cover"
                      src="https://lh3.googleusercontent.com/aida-public/AB6AXuAJiIRXYX-3Xir4i7i6igKSQQ67lM9-pSu4Z8LmnYu9iNcGWBhcjzJckudN_73g5dhoqqwVWHxGtkC96FsA3CDHjj4G3KaoSTCL9uvSC-slGYOoqCc4Y2M2RDp7rbDrsfzw5fmN2JnsFtpxu6EQSNlFTxaYFCkhYixMfXpuzzK8R71nzBIUYDZA5vk648RHEJWDbRHHhKNZxpTGAf8E52HNGbsEv43rQQa4zWm4hdZytpYUx3JjqXo2jOIt2GfvriRo9iN7MXiVEWeZ"
                    />
                    <button className="absolute bottom-0 right-0 size-7 bg-primary text-white rounded-full flex items-center justify-center shadow-md hover:scale-110 transition-transform">
                      <span className="material-symbols-outlined text-[16px]">photo_camera</span>
                    </button>
                  </div>
                  <div className="flex-1">
                    <p className="font-bold text-slate-800">Professor Lambda</p>
                    <p className="text-sm text-slate-500">lambda.admin@matrixsolve.ai</p>
                    <button className="mt-2 text-xs font-bold text-primary hover:underline">Change Password</button>
                  </div>
                </div>
                <div className="pt-6 border-t border-slate-100">
                  <p className="font-bold text-slate-800 mb-4">Data Management</p>
                  <div className="flex flex-wrap gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-xs font-bold text-slate-700 transition-colors">
                      <span className="material-symbols-outlined text-[18px]">download</span>
                      Export All Matrices (JSON)
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg text-xs font-bold text-slate-700 transition-colors">
                      <span className="material-symbols-outlined text-[18px]">history</span>
                      Clear Search History
                    </button>
                  </div>
                </div>
              </div>
            </section>
            <div className="flex items-center justify-between pt-4 pb-12">
              <button className="flex items-center gap-2 text-slate-400 hover:text-red-500 text-sm font-bold transition-colors">
                <span className="material-symbols-outlined text-[18px]">restart_alt</span>
                Reset to Defaults
              </button>
              <div className="flex gap-4">
                <button className="px-6 py-2.5 rounded-xl text-sm font-bold text-slate-600 hover:bg-slate-100 transition-colors">Cancel</button>
                <button className="px-8 py-2.5 bg-primary text-white rounded-xl text-sm font-bold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all active:scale-95">Save Changes</button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

