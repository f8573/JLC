import React, { useEffect, useState } from 'react'
import MainPage from './pages/MainPage'
import MatrixBasicPage from './pages/MatrixBasicPage'
import MatrixSpectralPage from './pages/MatrixSpectralPage'
import MatrixDecomposePage from './pages/MatrixDecomposePage'
import MatrixStructurePage from './pages/MatrixStructurePage'
import MatrixReportPage from './pages/MatrixReportPage'
import RecentPage from './pages/RecentPage'
import FavoritesPage from './pages/FavoritesPage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import DocumentationPage from './pages/DocumentationPage'
import { ThemeProvider } from './context/ThemeContext'
import { SettingsProvider } from './context/SettingsContext'

/**
 * Root application component that handles lightweight client-side routing.
 *
 * The router is implemented with `window.location` and `popstate` to keep the
 * build dependency-free while still supporting deep links into matrix analysis
 * sections.
 */
function App() {
  const [path, setPath] = useState(window.location.pathname || '/')

  useEffect(() => {
    function onPop() { setPath(window.location.pathname || '/') }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])
  if (path === '/favorites') return <FavoritesPage />
  if (path === '/history') return <HistoryPage />
  if (path === '/recent') return <RecentPage />
  if (path === '/settings') return <SettingsPage />
  if (path === '/documentation') return <DocumentationPage />
  if (path.startsWith('/matrix=')) {
    const match = path.match(/^\/matrix=([^/]+)(?:\/(basic|spectral|decompose|structure|report))?$/)
    if (match) {
      const matrixString = decodeURIComponent(match[1])
      const section = match[2] || 'basic'
      if (section === 'spectral') return <MatrixSpectralPage matrixString={matrixString} />
      if (section === 'decompose') return <MatrixDecomposePage matrixString={matrixString} />
      if (section === 'structure') return <MatrixStructurePage matrixString={matrixString} />
      if (section === 'report') return <MatrixReportPage matrixString={matrixString} />
      return <MatrixBasicPage matrixString={matrixString} />
    }
  }

  return <MainPage />
}

function AppWrapper() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <App />
      </SettingsProvider>
    </ThemeProvider>
  )
}

export default AppWrapper
