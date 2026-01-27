import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
// KaTeX styles for math rendering (requires `npm install katex`)
import 'katex/dist/katex.min.css'

/**
 * Application entry point.
 * Initializes the React root and mounts the App inside StrictMode.
 */
createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
