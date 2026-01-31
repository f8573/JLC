import React from 'react'
import MatrixLatex from '../matrix/MatrixLatex'
import Latex from './Latex'
import { ACCURACY_COLORS, computeAccuracySeverity, computeResidualMatrix, computeElementErrors } from '../../utils/accuracySeverity'
import { formatNumber } from '../../utils/format'

/**
 * Modal component showing detailed decomposition accuracy information.
 *
 * @param {Object} props
 * @param {boolean} props.open - Whether modal is open
 * @param {Function} props.onClose - Close handler
 * @param {string} props.title - Decomposition title
 * @param {string} props.formula - Decomposition formula (e.g., "A = QR")
 * @param {Object} props.validation - Validation object from API
 * @param {number[][]} props.originalMatrix - Original matrix A
 * @param {number[][]} props.reconstructedMatrix - Reconstructed matrix A'
 */
export default function DecompositionDetailModal({
  open,
  onClose,
  title,
  formula,
  validation,
  originalMatrix,
  reconstructedMatrix
}) {
  if (!open) return null

  const severity = computeAccuracySeverity(validation)
  const residual = computeResidualMatrix(originalMatrix, reconstructedMatrix)
  const { errors, severities } = computeElementErrors(originalMatrix, reconstructedMatrix)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto">
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose}></div>
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-4xl m-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-slate-200 px-6 py-4 rounded-t-2xl z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-bold text-slate-800">{title}</h2>
              {formula && (
                <span className="text-sm font-semibold px-3 py-1 rounded-full bg-primary/5 text-primary border border-primary/10">
                  <Latex tex={formula} />
                </span>
              )}
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-100 rounded-full transition-colors"
            >
              <span className="material-symbols-outlined text-slate-500">close</span>
            </button>
          </div>
          
          {/* Accuracy Summary Bar */}
          <div 
            className="mt-4 px-4 py-3 rounded-xl flex items-center justify-between"
            style={{ 
              backgroundColor: `${severity.color}15`,
              borderLeft: `4px solid ${severity.color}`
            }}
          >
            <div className="flex items-center gap-3">
              <span className="text-2xl">{severity.shortLabel.replace(/[^\p{Emoji}]/gu, '')}</span>
              <div>
                <div className="font-bold" style={{ color: severity.color }}>{severity.label}</div>
                <div className="text-xs text-slate-500">
                  Norm: {mapLevelName(validation?.normLevel)} | Element: {mapLevelName(validation?.elementLevel)}
                </div>
              </div>
            </div>
            <div className="text-right text-sm">
              <div className="font-mono text-slate-600">
                ‖R‖/‖A‖ = {formatNumber(validation?.normResidual, 2)}
              </div>
              <div className="font-mono text-slate-600">
                max|r_ij| = {formatNumber(validation?.elementResidual, 2)}
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Diagnostic Message */}
          {validation?.message && (
            <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
              <h3 className="text-sm font-bold text-slate-700 mb-2 flex items-center gap-2">
                <span className="material-symbols-outlined text-lg">info</span>
                Diagnostic Details
              </h3>
              <pre className="text-xs text-slate-600 whitespace-pre-wrap font-mono leading-relaxed">
                {validation.message}
              </pre>
            </div>
          )}

          {/* Reconstruction Visualization */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Original Matrix */}
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-slate-700 text-center">
                Original Matrix <Latex tex="A" />
              </h3>
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 flex justify-center overflow-x-auto">
                {originalMatrix ? (
                  <MatrixLatex data={originalMatrix} className="text-sm" />
                ) : (
                  <span className="text-xs text-slate-400">Not available</span>
                )}
              </div>
            </div>

            {/* Reconstructed Matrix */}
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-slate-700 text-center">
                Reconstruction <Latex tex="A' = " />{formula?.split('=')[1]?.trim() || '?'}
              </h3>
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 flex justify-center overflow-x-auto">
                {reconstructedMatrix ? (
                  <MatrixLatex data={reconstructedMatrix} className="text-sm" />
                ) : (
                  <span className="text-xs text-slate-400">Not available</span>
                )}
              </div>
            </div>

            {/* Residual Matrix */}
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-slate-700 text-center">
                Residual <Latex tex="R = A - A'" />
              </h3>
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 flex justify-center overflow-x-auto">
                {residual ? (
                  <ColoredResidualMatrix residual={residual} severities={severities} />
                ) : (
                  <span className="text-xs text-slate-400">Not available</span>
                )}
              </div>
            </div>
          </div>

          {/* Element-wise Error Heatmap */}
          {severities && (
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                <span className="material-symbols-outlined text-lg">grid_on</span>
                Element-wise Error Analysis
              </h3>
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                <ErrorHeatmap errors={errors} severities={severities} />
              </div>
              <ErrorLegend />
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-slate-50 border-t border-slate-200 px-6 py-4 rounded-b-2xl">
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-400">
              {validation?.passes ? '✓ Validation passes' : '✗ Validation failed'}
              {validation?.shouldWarn && ' (with warnings)'}
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-primary text-white rounded-lg font-semibold text-sm hover:bg-primary/90 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Render a residual matrix with colored entries based on severity.
 */
function ColoredResidualMatrix({ residual, severities }) {
  if (!residual || !severities) return null

  return (
    <div className="inline-block">
      <div className="flex items-center gap-1">
        <span className="text-slate-400 text-lg">[</span>
        <div className="space-y-0.5">
          {residual.map((row, i) => (
            <div key={i} className="flex gap-2">
              {row.map((val, j) => {
                const sev = severities[i]?.[j] || 'safe'
                return (
                  <span
                    key={j}
                    className="font-mono text-xs px-1 py-0.5 rounded"
                    style={{
                      backgroundColor: `${ACCURACY_COLORS[sev]}20`,
                      color: ACCURACY_COLORS[sev],
                      fontWeight: sev !== 'safe' ? 600 : 400
                    }}
                    title={`Error severity: ${sev}`}
                  >
                    {formatNumber(val, 2)}
                  </span>
                )
              })}
            </div>
          ))}
        </div>
        <span className="text-slate-400 text-lg">]</span>
      </div>
    </div>
  )
}

/**
 * Error heatmap visualization.
 */
function ErrorHeatmap({ errors, severities }) {
  if (!errors || !severities) return null

  const rows = errors.length
  const cols = errors[0]?.length || 0
  const cellSize = Math.max(24, Math.min(48, 300 / Math.max(rows, cols)))

  return (
    <div className="overflow-x-auto">
      <div 
        className="inline-grid gap-0.5"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`
        }}
      >
        {errors.map((row, i) =>
          row.map((err, j) => {
            const sev = severities[i]?.[j] || 'safe'
            return (
              <div
                key={`${i}-${j}`}
                className="flex items-center justify-center text-[9px] font-mono rounded"
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: ACCURACY_COLORS[sev],
                  color: sev === 'safe' || sev === 'mild' ? 'white' : 'rgba(0,0,0,0.8)'
                }}
                title={`[${i},${j}] error: ${formatNumber(err, 4)} (${sev})`}
              >
                {cellSize >= 32 ? formatNumber(err, 1) : ''}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

/**
 * Legend for error severity colors.
 */
function ErrorLegend() {
  const levels = [
    { level: 'safe', label: 'Excellent/Good' },
    { level: 'mild', label: 'Acceptable' },
    { level: 'moderate', label: 'Warning' },
    { level: 'severe', label: 'Poor' },
    { level: 'critical', label: 'Critical' }
  ]

  return (
    <div className="flex flex-wrap items-center gap-4 text-xs">
      <span className="text-slate-500 font-medium">Legend:</span>
      {levels.map(({ level, label }) => (
        <div key={level} className="flex items-center gap-1.5">
          <div
            className="w-3 h-3 rounded"
            style={{ backgroundColor: ACCURACY_COLORS[level] }}
          />
          <span className="text-slate-600">{label}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Map AccuracyLevel name to friendly display.
 */
function mapLevelName(level) {
  if (!level) return '—'
  return level.charAt(0) + level.slice(1).toLowerCase()
}
