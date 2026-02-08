import React, { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext()

const DEFAULT_ACCENT = '#7c3aed'
const DEFAULT_THEME = 'light'

/**
 * Theme provider that manages dark mode and accent color across the app.
 * Persists settings to localStorage and applies CSS variables.
 */
export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || DEFAULT_THEME
  })
  
  const [accentColor, setAccentColorState] = useState(() => {
    const saved = localStorage.getItem('accentColor')
    return saved || DEFAULT_ACCENT
  })

  // Apply theme class to document
  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    root.classList.toggle('light', theme !== 'dark')
    
    // Update CSS variables for accent color (used in custom styles)
    root.style.setProperty('--accent-color', accentColor)
    
    // Calculate lighter/darker variants
    const hex = accentColor.replace('#', '')
    const r = parseInt(hex.substr(0, 2), 16)
    const g = parseInt(hex.substr(2, 2), 16)
    const b = parseInt(hex.substr(4, 2), 16)
    
    // Lighter variant (for backgrounds)
    const lighterR = Math.min(255, r + Math.round((255 - r) * 0.9))
    const lighterG = Math.min(255, g + Math.round((255 - g) * 0.9))
    const lighterB = Math.min(255, b + Math.round((255 - b) * 0.9))
    const lighterColor = `rgb(${lighterR}, ${lighterG}, ${lighterB})`
    root.style.setProperty('--accent-light', lighterColor)
    
    // Darker variant (for hover)
    const darkerR = Math.round(r * 0.85)
    const darkerG = Math.round(g * 0.85)
    const darkerB = Math.round(b * 0.85)
    const darkerColor = `rgb(${darkerR}, ${darkerG}, ${darkerB})`
    root.style.setProperty('--accent-hover', darkerColor)
    
    // Re-configure Tailwind with new colors and force rebuild
    if (typeof tailwind !== 'undefined' && tailwind.config) {
      tailwind.config.theme.extend.colors.primary = accentColor
      tailwind.config.theme.extend.colors['primary-hover'] = darkerColor
      tailwind.config.theme.extend.colors['purple-light'] = lighterColor
      tailwind.config.theme.extend.colors['primary-light'] = lighterColor
      
      // Force Tailwind CDN to rebuild styles by triggering a config change
      const configScript = document.getElementById('tailwind-config')
      if (configScript) {
        const newConfig = `tailwind.config = ${JSON.stringify(tailwind.config)}`
        // Dispatch event to trigger rebuild
        window.dispatchEvent(new CustomEvent('tailwind-config-changed'))
      }
    }
  }, [theme, accentColor])

  // Remove first-load transition/animation suppression once initial paint is complete.
  useEffect(() => {
    const root = document.documentElement
    if (!root.classList.contains('no-theme-fade')) return
    const id = window.requestAnimationFrame(() => {
      root.classList.remove('no-theme-fade')
    })
    return () => window.cancelAnimationFrame(id)
  }, [])

  const setTheme = (newTheme) => {
    setThemeState(newTheme)
    localStorage.setItem('theme', newTheme)
  }

  const setAccentColor = (color) => {
    setAccentColorState(color)
    localStorage.setItem('accentColor', color)
  }

  const resetDefaults = () => {
    setTheme(DEFAULT_THEME)
    setAccentColor(DEFAULT_ACCENT)
  }

  return (
    <ThemeContext.Provider value={{ 
      theme, 
      setTheme, 
      accentColor, 
      setAccentColor,
      resetDefaults,
      isDark: theme === 'dark'
    }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}
