import React from 'react'
import { navigate } from '../../utils/navigation'

const tabs = [
  { id: 'basic', label: 'Basic Properties' },
  { id: 'spectral', label: 'Spectral Analysis' },
  { id: 'decompose', label: 'Matrix Decompositions' },
  { id: 'structure', label: 'Structure' }
]

/**
 * Tab navigation for matrix analysis sections.
 *
 * @param {Object} props
 * @param {string} props.matrixString
 * @param {'basic'|'spectral'|'decompose'|'structure'} [props.activeTab='basic']
 */
export default function MatrixTabs({ matrixString, activeTab = 'basic' }) {
  const encoded = encodeURIComponent(matrixString || '')
  return (
    <div className="flex border-b border-slate-200 bg-slate-50/50">
      {tabs.map((tab) => {
        const href = `/matrix=${encoded}/${tab.id}`
        const active = tab.id === activeTab
        return (
          <a
            key={tab.id}
            href={href}
            onClick={(e) => { e.preventDefault(); navigate(href) }}
            className={`flex-1 px-4 py-4 text-xs font-bold uppercase tracking-widest border-b-2 transition-all ${
              active ? 'tab-active' : 'tab-inactive'
            }`}
          >
            {tab.label}
          </a>
        )
      })}
    </div>
  )
}

