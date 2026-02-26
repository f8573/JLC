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

  // Apply theme class and accent variables to document root.
  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
    
    // Update CSS variables for accent color (used in custom styles)
    root.style.setProperty('--accent-color', accentColor)
    
    // Calculate lighter/darker variants
    const hex = accentColor.replace('#', '')
    const r = parseInt(hex.slice(0, 2), 16)
    const g = parseInt(hex.slice(2, 4), 16)
    const b = parseInt(hex.slice(4, 6), 16)
    
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
    
  }, [theme, accentColor])

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
