import React from 'react'
import Button from '../ui/Button'

export default function MatrixActions({ onAnalyze, onTranspose }) {
  return (
    <div className="flex flex-col sm:flex-row items-center justify-center gap-5 w-full max-w-md">
      <Button 
        onClick={onAnalyze} 
        variant="primary" 
        icon="analytics"
        className="flex-1"
      >
        Analyze Matrix
      </Button>
      <Button 
        onClick={onTranspose} 
        variant="secondary"
      >
        <span className="material-symbols-outlined text-xl group-hover:rotate-90 transition-transform duration-300">
          swap_horiz
        </span>
        Transpose
      </Button>
    </div>
  )
}
