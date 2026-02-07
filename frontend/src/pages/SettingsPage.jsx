import React, { useState, useEffect } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { useMatrixCompute } from '../hooks/useMatrixCompute'
import { useTheme } from '../context/ThemeContext'
import { useSettings } from '../context/SettingsContext'
import pkg from '../../package.json'

/**
 * User settings and preferences view.
 */
export default function SettingsPage() {
  const handleCompute = useMatrixCompute()
  const { theme: globalTheme, accentColor: globalAccent, setTheme: applyTheme, setAccentColor: applyAccent, resetDefaults: resetGlobalDefaults } = useTheme()
  const { 
    clearSearchHistory,
    exportAllMatrices
  } = useSettings()
  
  // Local settings state (for preview before saving)
  const [theme, setTheme] = useState(globalTheme)
  const [accentColor, setAccentColor] = useState(globalAccent)
  
  // Sync local state with global when they change externally
  useEffect(() => {
    setTheme(globalTheme)
    setAccentColor(globalAccent)
  }, [globalTheme, globalAccent])
  
  const accentColors = ['#7c3aed', '#9333ea', '#a855f7', '#c084fc', '#6366f1']
  
  const handleResetDefaults = () => {
    setTheme('light')
    setAccentColor('#7c3aed')
    // Apply reset immediately
    resetGlobalDefaults()
    // global precision/output format handled by resetGlobalDefaults where applicable
  }
  
  const handleSaveChanges = () => {
    // Apply theme and accent color globally
    applyTheme(theme)
    applyAccent(accentColor)
    // Show brief confirmation (optional - could add toast notification)
    window.history.back()
  }

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all search history? This action cannot be undone.')) {
      clearSearchHistory()
      // Show brief notification
      alert('Search history cleared successfully.')
    }
  }

  const handleExportMatrices = () => {
    const result = exportAllMatrices()
    if (result.success) {
      alert(`Successfully exported ${result.count} unique matrices.`)
    } else {
      alert(`Failed to export matrices: ${result.error}`)
    }
  }

  const [serverVersion, setServerVersion] = useState('unknown')
  const frontendVersion = pkg?.version || 'unknown'

  useEffect(() => {
    let mounted = true
    fetch('/api/ping')
      .then(res => res.json())
      .then(data => {
        if (!mounted) return
        if (data && data.version) setServerVersion(String(data.version))
      })
      .catch(() => {
        if (mounted) setServerVersion('unavailable')
      })
    return () => { mounted = false }
  }, [])
  
  return (
    <div className="bg-background-light dark:bg-slate-900 font-display text-slate-900 dark:text-slate-100 h-screen overflow-hidden transition-colors duration-300">
      <Header inputValue="" onCompute={handleCompute} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="settings" />
        <main className="flex-1 overflow-y-auto custom-scrollbar bg-background-light dark:bg-slate-900 transition-colors duration-300">
          <div className="max-w-[800px] mx-auto p-8 space-y-10">
            <div className="space-y-1">
              <h1 className="text-3xl font-black tracking-tight text-slate-900 dark:text-white uppercase">Settings</h1>
              <p className="text-slate-500 dark:text-slate-400 text-sm">Configure your personal preferences and computational environment.</p>
            </div>
            <section className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden transition-colors duration-300">
              <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
                <h2 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[18px]">palette</span>
                  Appearance
                </h2>
              </div>
              <div className="p-6 space-y-8">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-bold text-slate-800 dark:text-slate-200">Theme Mode</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Switch between light and dark visual interfaces</p>
                  </div>
                  <div className="flex items-center bg-slate-100 dark:bg-slate-700 p-1 rounded-xl">
                    <button 
                      onClick={() => setTheme('light')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-lg transition-all ${theme === 'light' ? 'bg-white dark:bg-slate-600 shadow-sm text-primary' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                    >
                      Light
                    </button>
                    <button 
                      onClick={() => setTheme('dark')}
                      className={`px-4 py-1.5 text-xs font-bold rounded-lg transition-all ${theme === 'dark' ? 'bg-white dark:bg-slate-600 shadow-sm text-primary' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                    >
                      Dark
                    </button>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="font-bold text-slate-800 dark:text-slate-200">Accent Color</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Choose your primary theme color (shades of purple)</p>
                  </div>
                  <div className="flex gap-4">
                    {accentColors.map((color, idx) => (
                      <button 
                        key={color}
                        onClick={() => setAccentColor(color)}
                        className={`size-10 rounded-full hover:scale-110 transition-transform ${accentColor === color ? 'ring-4 ring-primary/20 border-2 border-white dark:border-slate-600' : ''}`}
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </section>
            {/* Computational Preferences removed per request */}
            <section className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 overflow-hidden transition-colors duration-300">
              <div className="px-6 py-4 border-b border-slate-100 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/50">
                <h2 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-[18px]">folder</span>
                  Data Management
                </h2>
              </div>
              <div className="p-6">
                <div className="flex flex-wrap gap-3">
                  <button 
                    onClick={handleExportMatrices}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 border border-slate-200 dark:border-slate-600 rounded-lg text-xs font-bold text-slate-700 dark:text-slate-200 transition-colors"
                  >
                    <span className="material-symbols-outlined text-[18px]">download</span>
                    Export All Matrices (JSON)
                  </button>
                  <button 
                    onClick={handleClearHistory}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-50 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 border border-slate-200 dark:border-slate-600 rounded-lg text-xs font-bold text-slate-700 dark:text-slate-200 transition-colors"
                  >
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
              <div className="text-xs text-slate-500 dark:text-slate-400 flex flex-col items-start">
                <div>Frontend: {frontendVersion}</div>
                <div>Server: {serverVersion}</div>
              </div>
              <div className="flex gap-4">
                <button 
                  onClick={() => window.history.back()}
                  className="px-6 py-2.5 rounded-xl text-sm font-bold text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                >Cancel</button>
                <button 
                  onClick={handleSaveChanges}
                  className="px-8 py-2.5 bg-primary text-white rounded-xl text-sm font-bold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all active:scale-95"
                >Save Changes</button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

