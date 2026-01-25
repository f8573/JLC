import React from 'react'
import NumberInput from '../ui/NumberInput'

export default function MatrixDimensionControl({ rows, cols, onDimensionChange }) {
  return (
    <div className="flex flex-col items-center gap-3 mb-10">
      <label className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
        Dimensions
      </label>
      <div className="flex items-center gap-2">
        <NumberInput
          label="rows"
          value={rows}
          onChange={(e) => onDimensionChange(e.target.value, cols)}
        />
        <span className="text-sm font-bold">×</span>
        <NumberInput
          label="cols"
          value={cols}
          onChange={(e) => onDimensionChange(rows, e.target.value)}
        />
      </div>
    </div>
  )
}
