import React from 'react'
import FeatureCard from './FeatureCard'

const features = [
  {
    icon: 'keyboard',
    title: 'Shortcuts',
    description: 'Use [Enter] to run analysis and [Arrow Keys] to navigate cells.'
  },
  {
    icon: 'auto_graph',
    title: 'Live Plotting',
    description: 'Visualizations for eigenvalues and row reduction are generated instantly.',
    hasBorder: true
  },
  {
    icon: 'terminal',
    title: 'API Access',
    description: 'Export computations to Python, MATLAB, or LaTeX formats.',
    hasBorder: true
  }
]

/**
 * Grid of feature callouts for the matrix input console.
 */
export default function FeatureGrid() {
  return (
    <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
      {features.map((feature, idx) => (
        <FeatureCard key={idx} {...feature} />
      ))}
    </div>
  )
}
