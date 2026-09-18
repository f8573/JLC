import React, { useEffect, useState } from 'react'
import katex from 'katex'

/**
 * Renders KaTeX math from a TeX string with a safe fallback.
 *
 * @param {Object} props
 * @param {string} [props.tex='']
 * @param {boolean} [props.displayMode=false]
 * @param {string} [props.className='']
 * @param {Object} [props.style={}]
 */
export default function Latex({ tex = '', displayMode = false, className = '', style = {} }) {
  const [html, setHtml] = useState('')

  useEffect(() => {
    if (!tex) {
      setHtml('')
      return
    }
    // KaTeX warns for some Unicode punctuation in strict mode.
    // Normalize common dash/minus variants to ASCII-safe TeX text.
    const normalizedTex = String(tex)
      .replace(/\u2014/g, '--') // em dash
      .replace(/\u2013/g, '--') // en dash
      .replace(/\u2212/g, '-')  // unicode minus
    try {
      const engine = (typeof katex.renderToString === 'function') ? katex : (katex.default || katex)
      const rendered = engine.renderToString(normalizedTex, { throwOnError: false, displayMode })
      setHtml(rendered)
    } catch (err) {
      // If rendering fails, keep plain text fallback and log error for debugging
      // eslint-disable-next-line no-console
      console.error('KaTeX render failed for tex:', normalizedTex, err)
      setHtml(null)
    }
  }, [tex, displayMode])

  if (html === null) {
    return <span className={className} style={style}>{tex}</span>
  }
  return <span className={className} style={style} dangerouslySetInnerHTML={{ __html: html }} />
}
