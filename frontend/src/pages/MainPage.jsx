import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import MatrixInput from '../components/MatrixInput'

/**
 * Landing page layout for the matrix explorer.
 * Renders the global header, navigation sidebar, and the matrix input panel.
 */
export default function MainPage() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#ffffff] dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans selection:bg-primary/20 transition-colors duration-300">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active="home" />
        <MatrixInput />
      </div>
    </div>
  )
}
