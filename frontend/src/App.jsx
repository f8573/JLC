import React, {useEffect, useState} from 'react'
import katex from 'katex/dist/katex.mjs'
import 'katex/dist/katex.min.css'

export default function App(){
  const [latexList, setLatexList] = useState(null)
  const [original, setOriginal] = useState(null)
  const [eigenvalues, setEigenvalues] = useState(null)
  const [error, setError] = useState(null)

  useEffect(()=>{
    const url = 'http://localhost:8080/api/latex'
    console.log('Fetching LaTeX from', url)
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status)
        return r.json()
      })
      .then(j=>{
        console.log('received', j)
        // Support both legacy `latex` array and new structured response
        setLatexList(j.schur || j.latex || [])
        setOriginal(j.original || null)
        setEigenvalues(j.eigenvalues || null)
      })
      .catch(e=>{
        console.error('fetch error', e)
        setError(e.message)
      })
  },[])

  return (
    <div style={{fontFamily:'Arial, sans-serif',padding:20}}>
      <h1>JLC Frontend — LaTeX Visualizer</h1>
      {error && <p style={{color:'crimson'}}>Error: {error}</p>}
      {!latexList && !error && <p>Loading LaTeX from backend...</p>}
      {latexList && latexList.length === 0 && <p>No Schur matrices returned.</p>}

      {original && (
        <div style={{marginBottom:20, padding:10, background:'#f7f7ff', border:'1px solid #eee'}}>
          <h3>Original Matrix</h3>
          <div dangerouslySetInnerHTML={{__html: katex.renderToString(original, {displayMode:true, throwOnError:false})}} />
        </div>
      )}

      {eigenvalues && (
        <div style={{marginBottom:20, padding:10, background:'#fffaf0', border:'1px solid #eee'}}>
          <h3>Eigenvalues</h3>
          <ul>
            {eigenvalues.map((ev, idx) => <li key={idx}>{ev}</li>)}
          </ul>
        </div>
      )}

      {latexList && latexList.map((math, i) => {
        let html = ''
        try {
          html = katex.renderToString(math, {displayMode:true, throwOnError:false})
        } catch (err) {
          console.error('KaTeX render error', err, math)
          html = `<pre style="white-space:pre-wrap">${math.replace(/</g,'&lt;')}</pre>`
        }
        return (
          <div key={i} style={{marginBottom:20, padding:10, background:'#fff', border:'1px solid #eee'}} dangerouslySetInnerHTML={{__html: html}} />
        )
      })}
      <p style={{marginTop:20}}>Dev setup: run the Java backend and this frontend dev server.</p>
    </div>
  )
}
