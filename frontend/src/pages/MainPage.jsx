import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import MatrixInput from '../components/MatrixInput'
import { useMatrixCompute } from '../hooks/useMatrixCompute'

/**
 * Landing page layout for the matrix explorer.
 * Renders the global header, navigation sidebar, and the matrix input panel.
 */
export default function MainPage() {
  const handleCompute = useMatrixCompute()

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#ffffff] dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans selection:bg-primary/20 transition-colors duration-300">
      <Header onCompute={handleCompute} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar active="home" />
        <MatrixInput />
      </div>
    </div>
  )
}
