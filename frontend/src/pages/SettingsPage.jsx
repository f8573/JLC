import React, { useState } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

/**
 * User settings and preferences view.
 */
export default function SettingsPage() {
  const handleCompute = useMatrixCompute()
  
  // Settings state
  const [theme, setTheme] = useState('light')
  const [accentColor, setAccentColor] = useState('#7c3aed')
  const [precision, setPrecision] = useState('6')
  const [outputFormat, setOutputFormat] = useState('numeric')
  
  const accentColors = ['#7c3aed', '#9333ea', '#a855f7', '#c084fc', '#6366f1']
  
  const handleResetDefaults = () => {
    setTheme('light')
    setAccentColor('#7c3aed')
    setPrecision('6')
    setOutputFormat('numeric')
  }
  
  return (
    <div className="bg-background-light font-display text-slate-900 h-screen overflow-hidden">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="settings" />
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
                    <button 
                      onClick={() => setTheme('light')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-lg ${theme === 'light' ? 'bg-white shadow-sm text-primary' : 'text-slate-400 hover:text-slate-600'}`}
                    >
                      Light
                    </button>
                    <button 
                      onClick={() => setTheme('dark')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-lg ${theme === 'dark' ? 'bg-white shadow-sm text-primary' : 'text-slate-400 hover:text-slate-600'}`}
                    >
                      Dark
                    </button>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="font-bold text-slate-800">Accent Color</p>
                    <p className="text-xs text-slate-500">Choose your primary theme color (shades of purple)</p>
                  </div>
                  <div className="flex gap-4">
                    {accentColors.map((color, idx) => (
                      <button 
                        key={color}
                        onClick={() => setAccentColor(color)}
                        className={`size-10 rounded-full hover:scale-110 transition-transform ${accentColor === color ? 'ring-4 ring-primary/20 border-2 border-white' : ''}`}
                        style={{ backgroundColor: color }}
                      />
                    ))}
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
                      value={precision}
                      onChange={(e) => setPrecision(e.target.value)}
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
                      <label className="flex-1 cursor-not-allowed">
                        <input disabled className="hidden peer" name="format" type="radio" value="symbolic" checked={outputFormat === 'symbolic'} onChange={() => {}} />
                        <div className="p-3 text-center rounded-xl border border-slate-200 bg-slate-50 text-slate-400 relative">
                          <span className="text-xs font-bold">Symbolic</span>
                          <span className="block text-[10px] text-slate-400 mt-1">Coming Soon...</span>
                        </div>
                      </label>
                      <label className="flex-1 cursor-pointer">
                        <input className="hidden peer" name="format" type="radio" value="numeric" checked={outputFormat === 'numeric'} onChange={() => setOutputFormat('numeric')} />
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
                  <span className="material-symbols-outlined text-primary text-[18px]">folder</span>
                  Data Management
                </h2>
              </div>
              <div className="p-6">
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
            </section>
            <div className="flex items-center justify-between pt-4 pb-12">
              <button 
                onClick={handleResetDefaults}
                className="flex items-center gap-2 text-slate-400 hover:text-red-500 text-sm font-bold transition-colors"
              >
                <span className="material-symbols-outlined text-[18px]">restart_alt</span>
                Reset to Defaults
              </button>
              <div className="flex gap-4">
                <button 
                  onClick={() => window.history.back()}
                  className="px-6 py-2.5 rounded-xl text-sm font-bold text-slate-600 hover:bg-slate-100 transition-colors"
                >Cancel</button>
                <button className="px-8 py-2.5 bg-primary text-white rounded-xl text-sm font-bold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all active:scale-95">Save Changes</button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

