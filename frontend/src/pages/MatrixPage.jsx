import React from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import MatrixResults from '../components/MatrixResults'

const DEFAULT_MATRIX_STRING = '[[4, 7, 2], [1, 0, 8], [3, 9, 5]]'

/**
 * High-level matrix dashboard page used for default analysis routing.
 *
 * @param {Object} props
 * @param {string} [props.matrixString] - Serialized matrix payload from the URL.
 */
export default function MatrixPage({ matrixString }) {
  const inputValue = matrixString && matrixString.trim().length > 0 ? matrixString : DEFAULT_MATRIX_STRING

  return (
    <div className="bg-background-light font-display text-slate-900 h-screen overflow-hidden">
      <Header inputValue={inputValue} />
      <div className="flex h-[calc(100vh-68px)] overflow-hidden">
        <Sidebar active="analysis" showCurrentAnalysis={true} />
        <MatrixResults />
      </div>
    </div>
  )
}
