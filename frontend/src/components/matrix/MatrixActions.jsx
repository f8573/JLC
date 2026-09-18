import React from 'react'
import Button from '../ui/Button'

/**
 * Action bar for analyzing and transposing matrices.
 *
 * @param {Object} props
 * @param {() => void} [props.onAnalyze]
 * @param {() => void} [props.onTranspose]
 * @param {() => void} [props.onFavorite]
 * @param {() => void} [props.onUpload]
 * @param {boolean} [props.uploadBusy]
 */
export default function MatrixActions({ onAnalyze, onTranspose, onFavorite, onUpload, uploadBusy = false }) {
  return (
    <div className="flex flex-col sm:flex-row flex-wrap items-center justify-center gap-5 w-full max-w-2xl">
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
      <Button
        onClick={onFavorite}
        variant="ghost"
      >
        <span className="material-symbols-outlined text-[20px] text-yellow-500">star</span>
        Favorite
      </Button>
      <Button
        onClick={onUpload}
        variant="secondary"
        disabled={uploadBusy}
      >
        <span className="material-symbols-outlined text-[20px]">upload_file</span>
        {uploadBusy ? 'Uploading...' : 'Upload File'}
      </Button>
    </div>
  )
}
