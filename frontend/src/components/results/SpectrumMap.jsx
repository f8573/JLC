import React, { useMemo, useState } from 'react'

function complexDist(a, b) {
  const dx = (a?.real ?? 0) - (b?.real ?? 0)
  const dy = (a?.imag ?? 0) - (b?.imag ?? 0)
  return Math.hypot(dx, dy)
}

function makeSeededRandom(seed) {
  let s = seed || 12345
  return function () {
    s = Math.imul(48271, s) % 2147483647
    return (s & 2147483647) / 2147483647
  }
}

export default function SpectrumMap({ eigenvalues = [], spectralRadius = 0, width = 180, height = 140, compact = false }) {
  const [tab, setTab] = useState('scatter')
  // allow a larger symmetric t-range for perturbation (center 0)
  const [t, setT] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)

  const view = useMemo(() => {
    // For consistent visuals, use a square view centered at 0 that includes the spectral radius
    let maxAbs = spectralRadius || 0
    for (const ev of eigenvalues) {
      const r = Math.abs(ev?.real ?? 0)
      const i = Math.abs(ev?.imag ?? 0)
      maxAbs = Math.max(maxAbs, r, i)
    }
    if (!isFinite(maxAbs) || maxAbs === 0) maxAbs = 1
    // small padding
    maxAbs = maxAbs * 1.12
    return { minR: -maxAbs, maxR: maxAbs, minI: -maxAbs, maxI: maxAbs }
  }, [eigenvalues, spectralRadius])

  const mapX = (r, w = width) => ((r - view.minR) / (view.maxR - view.minR)) * w
  const mapY = (i, h = height) => h - ((i - view.minI) / (view.maxI - view.minI)) * h

  // simple trajectories: deterministic pseudo-random direction per eigenvalue
  // avoid rendering extremely large numbers of SVG elements: sample eigenvalues for trajectories
  const MAX_TRAJ = 200
  const sampledIndices = useMemo(() => {
    if (!eigenvalues || eigenvalues.length <= MAX_TRAJ) return eigenvalues.map((_, i) => i)
    const step = Math.ceil(eigenvalues.length / MAX_TRAJ)
    const idxs = []
    for (let i = 0; i < eigenvalues.length; i += step) idxs.push(i)
    return idxs
  }, [eigenvalues])

  const trajectories = useMemo(() => {
    const rnd = makeSeededRandom(42)
    return sampledIndices.map((idx) => {
      const ev = eigenvalues[idx]
      const angle = rnd() * Math.PI * 2
      const mag = (Math.abs((ev?.real ?? 0)) + Math.abs((ev?.imag ?? 0)) + 1) * 0.04
      const dx = Math.cos(angle) * mag
      const dy = Math.sin(angle) * mag
      const points = []
      const steps = 24
      // generate points for tt in [-1, 1] to allow visible back-and-forth trajectories
      for (let s = -steps; s <= steps; s++) {
        const tt = s / steps
        points.push({ real: (ev?.real ?? 0) + dx * tt, imag: (ev?.imag ?? 0) + dy * tt })
      }
      return points
    })
  }, [eigenvalues, sampledIndices])

  // pseudospectrum approximation: color by min distance to eigenvalues
  const grid = useMemo(() => {
    // lower resolution grid when many eigenvalues to reduce SVG load
    const cols = eigenvalues && eigenvalues.length > 300 ? 60 : 100
    const rows = cols
    const cells = []
    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        const gx = view.minR + (cx + 0.5) * (view.maxR - view.minR) / cols
        const gy = view.minI + (cy + 0.5) * (view.maxI - view.minI) / rows
        let minD = Infinity
        for (const ev of eigenvalues) minD = Math.min(minD, complexDist({ real: gx, imag: gy }, ev))
        cells.push({ gx, gy, d: minD, cx, cy })
      }
    }
    return { cols, rows, cells }
  }, [view, eigenvalues])

  const renderSVG = (w, h) => (
    <svg width={w} height={h}>
      {/* pseudospectrum grid */}
      {tab === 'pseudo' && grid.cells.map((c, i) => {
        // map distance to color (closer -> hotter)
        const val = Math.exp(-Math.max(0, 3 * (c.d)))
        const r = Math.round(255 * (1 - val * 0.6))
        const g = Math.round(220 * (1 - val * 0.3))
        const b = Math.round(210 * (1 - val * 0.1))
        return <rect key={i} x={ (c.cx * w) / grid.cols } y={ (c.cy * h) / grid.rows } width={w / grid.cols + 1} height={h / grid.rows + 1} fill={`rgb(${r},${g},${b})`} />
      })}

      {/* spectral radius circle (scatter) */}
      {tab === 'scatter' && (
        <circle cx={mapX(0, w)} cy={mapY(0, h)} r={ (spectralRadius / (view.maxR - view.minR)) * w } fill="none" stroke="#FF8A00" strokeWidth={1.5} opacity={0.9} />
      )}

      {/* eigenvalue points: show in scatter tab or in compact preview */}
      {(tab === 'scatter' || compact) && eigenvalues.map((ev, i) => (
        <g key={i}>
          <circle cx={mapX(ev?.real ?? 0, w)} cy={mapY(ev?.imag ?? 0, h)} r={4} fill="#2563eb" stroke="#1e40af" strokeWidth={0.6} />
        </g>
      ))}

      {/* trajectories */}
      {tab === 'traj' && trajectories.map((pts, idx) => (
        <g key={`traj-${idx}`}>
          <polyline
            fill="none"
            stroke={idx % 2 === 0 ? '#7c3aed' : '#0ea5a6'}
            strokeWidth={1}
            strokeOpacity={0.9}
            points={pts.map(p => `${mapX(p.real, w)},${mapY(p.imag, h)}`).join(' ')}
          />
          {/* marker at t */}
          {(() => {
            // map t in [-100,100] to normalized [-1,1] then to index [0, pts.length-1]
            const norm = Math.max(-1, Math.min(1, (t || 0) / 100))
            const k = Math.max(0, Math.min(pts.length - 1, Math.floor((norm + 1) / 2 * (pts.length - 1))))
            const p = pts[k] || pts[0]
            return <circle cx={mapX(p.real, w)} cy={mapY(p.imag, h)} r={3.5} fill={idx % 2 === 0 ? '#7c3aed' : '#0ea5a6'} />
          })()}
        </g>
      ))}

      {/* axes */}
      <line x1={0} y1={mapY(0, h)} x2={w} y2={mapY(0, h)} stroke="#e6e6e6" strokeWidth={1} />
      <line x1={mapX(0, w)} y1={0} x2={mapX(0, w)} y2={h} stroke="#e6e6e6" strokeWidth={1} />
    </svg>
  )

  return (
    <div className="w-full relative">
      {/* show tabs only when not compact preview; move tabs into fullscreen modal for enlarged view */}
      {(!compact && !fullscreen) && (
        <div className="flex items-center justify-between mb-2">
          <div className="flex gap-2">
            <button className={`px-2 py-1 text-xs rounded ${tab === 'scatter' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('scatter')}>Spectrum</button>
            <button className={`px-2 py-1 text-xs rounded ${tab === 'traj' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('traj')}>Trajectories</button>
            <button className={`px-2 py-1 text-xs rounded ${tab === 'pseudo' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('pseudo')}>Pseudospectrum</button>
          </div>
          {/* slider removed; `t` ranges -100..100 and is controlled externally or programmatically */}
        </div>
      )}
      <div style={{ width, height }} className="rounded border border-slate-100 bg-white overflow-hidden cursor-pointer" onClick={() => setFullscreen(true)}>
        {compact ? (
          // compact preview: spectral radius circle + eigenvalues (blue dots)
          <svg width={width} height={height}>
            <circle cx={mapX(0, width)} cy={mapY(0, height)} r={(spectralRadius / (view.maxR - view.minR)) * width} fill="none" stroke="#FF8A00" strokeWidth={1.5} opacity={0.9} />
            {eigenvalues.map((ev, i) => (
              <circle key={i} cx={mapX(ev?.real ?? 0, width)} cy={mapY(ev?.imag ?? 0, height)} r={4} fill="#2563eb" stroke="#1e40af" strokeWidth={0.6} />
            ))}
          </svg>
        ) : renderSVG(width, height)}
      </div>

      {fullscreen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setFullscreen(false)}>
          <div className="bg-white rounded shadow-lg p-4" onClick={(e) => e.stopPropagation()}>
              {/* Tabs appear inside the modal so they are clickable */}
              <div className="mb-2">
                <div className="flex items-center justify-between">
                  <div className="flex gap-2">
                    <button className={`px-2 py-1 text-xs rounded ${tab === 'scatter' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('scatter')}>Spectrum</button>
                    <button className={`px-2 py-1 text-xs rounded ${tab === 'traj' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('traj')}>Trajectories</button>
                    <button className={`px-2 py-1 text-xs rounded ${tab === 'pseudo' ? 'bg-primary text-white' : 'bg-slate-50'}`} onClick={() => setTab('pseudo')}>Pseudospectrum</button>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* slider removed in fullscreen modal as well */}
                    <button className="px-2 py-1 text-sm bg-slate-50 rounded" onClick={() => setFullscreen(false)}>Close</button>
                  </div>
                </div>
              </div>
              {renderSVG(Math.min(window.innerWidth * 0.8, 900), Math.min(window.innerWidth * 0.8, 900))}
          </div>
        </div>
      )}
    </div>
  )
}
