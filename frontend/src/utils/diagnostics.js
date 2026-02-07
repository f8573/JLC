/**
 * Parse a serialized matrix string into a 2D array.
 *
 * @param {string} matrixString
 * @returns {number[][] | null}
 */
export function parseMatrixString(matrixString) {
  if (!matrixString) return null
  try {
    const parsed = JSON.parse(matrixString)
    if (!Array.isArray(parsed)) return null
    return parsed
  } catch {
    return null
  }
}

/**
 * Convert grid string values into numeric matrix data.
 *
 * @param {string[][]} values
 * @returns {number[][]}
 */
export function valuesToMatrixData(values) {
  return values.map((row) =>
    row.map((value) => {
      const num = Number(value)
      return Number.isFinite(num) ? num : 0
    })
  )
}

/**
 * Serialize a numeric matrix into a JSON string.
 *
 * @param {number[][]} matrixData
 * @returns {string}
 */
export function matrixToString(matrixData) {
  return JSON.stringify(matrixData)
}

/**
 * Build a namespaced cache key for a matrix string.
 *
 * @param {string} matrixString
 * @returns {string}
 */
function cacheKey(matrixString) {
  return `diagnostics:${encodeURIComponent(matrixString)}`
}

/**
 * Load cached diagnostics from session storage.
 *
 * @param {string} matrixString
 * @returns {any | null}
 */
export function loadCachedDiagnostics(matrixString) {
  if (!matrixString) return null
  try {
    const raw = sessionStorage.getItem(cacheKey(matrixString))
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed?.data ?? null
  } catch {
    return null
  }
}

/**
 * Persist diagnostics to session storage.
 *
 * @param {string} matrixString
 * @param {any} diagnostics
 */
function cacheDiagnostics(matrixString, diagnostics) {
  if (!matrixString || !diagnostics) return
  try {
    sessionStorage.setItem(
      cacheKey(matrixString),
      JSON.stringify({
        ts: Date.now(),
        data: diagnostics
      })
    )
  } catch {
    // ignore cache failures
  }
}

/**
 * Fetch diagnostics from the backend API.
 *
 * @param {number[][]} matrixData
 * @returns {Promise<any>}
 */
async function fetchDiagnostics(matrixData) {
  const response = await fetch('/api/diagnostics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ matrix: matrixData })
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || 'Diagnostics request failed')
  }

  const data = await response.json()
  return normalizeDiagnostics(data)
}

/**
 * Fetch diagnostics and update cache.
 *
 * @param {number[][]} matrixData
 * @param {string} matrixString
 * @returns {Promise<any>}
 */
export async function analyzeAndCache(matrixData, matrixString) {
  const diagnostics = await fetchDiagnostics(matrixData)
  cacheDiagnostics(matrixString, diagnostics)
  return diagnostics
}

/**
 * Normalize API diagnostics into a flat, backwards-compatible shape.
 *
 * @param {any} resp
 * @returns {any}
 */
function normalizeDiagnostics(resp) {
  if (!resp || typeof resp !== 'object') return resp

  // If already flat (no grouped keys), return as-is
  const topKeys = Object.keys(resp)
  const hasGroups = topKeys.some(k => ['basicProperties', 'spectralAnalysis', 'structuralProperties', 'matrixDecompositions'].includes(k))
  if (!hasGroups) return resp

  const flat = {}

  function tryAssign(name, val) {
    if (val && typeof val === 'object' && 'value' in val) {
      flat[name] = val.value
    } else {
      flat[name] = val
    }
  }

  // Flatten up to three levels: group -> section -> item
  for (const [, group] of Object.entries(resp)) {
    if (!group || typeof group !== 'object') continue
    for (const [sectionName, section] of Object.entries(group)) {
      if (section && typeof section === 'object') {
        // Keep entire decomposition objects intact at top level
        if (sectionName === 'primaryDecompositions' || sectionName === 'similarityAndSpectral' || 
            sectionName === 'derivedMatrices' || sectionName === 'subspaceBases') {
          for (const [k, v] of Object.entries(section)) {
            if (flat[k] === undefined) {
              flat[k] = v
            }
          }
        } else {
          for (const [k, v] of Object.entries(section)) {
            // prefer existing top-level keys (do not overwrite)
            if (flat[k] === undefined) {
              tryAssign(k, v)
            }
            // if v looks like a DiagnosticItem with a nested matrix/value object
            if (v && typeof v === 'object' && 'value' in v && v.value && typeof v.value === 'object') {
              // merge inner fields (e.g., qr -> { q, r })
              for (const [innerK, innerV] of Object.entries(v.value)) {
                if (flat[innerK] === undefined) {
                  flat[innerK] = innerV
                }
              }
            }
          }
        }
      } else {
        // section is primitive
        if (flat[sectionName] === undefined) tryAssign(sectionName, section)
      }
    }
  }

  // Also keep the original grouped response under a key for advanced uses
  flat._raw = resp
  // Build a backward-compatible per-eigenvalue aggregated object if the backend returned
  // separate per-eigen lists (or an older nested structure). This constructs
  // `eigenInformationPerValue` so the frontend can consume a single canonical shape.
  try {
    if (flat.eigenInformationPerValue === undefined) {
      const rawEigenInfo = resp?.spectralAnalysis?.eigenInformation
      const esList = rawEigenInfo?.eigenspacePerEigenvalue || rawEigenInfo?.eigenspacePerEigenvalue || null
      const ebList = rawEigenInfo?.eigenbasisPerEigenvalue || null
      const ev = flat.eigenvalues || null
      const alg = flat.algebraicMultiplicity || null
      const geom = flat.geometricMultiplicity || null
        if (esList && Array.isArray(esList) && esList.length > 0) {
          const per = []
          for (let i = 0; i < esList.length; i++) {
            const es = esList[i]
            const eb = ebList && ebList[i] ? ebList[i] : null
            // Prefer vector-matching when the backend returned a compact per-eigen list
            // (i.e. esList length != full eigenvalues length). Only use direct
            // index mapping when lengths match.
            let eigen = null
            if (ev && Array.isArray(ev) && ev.length === esList.length && ev[i]) eigen = ev[i]
            const vectors = es && es.vectors ? es.vectors : null
            const basisVectors = eb && eb.vectors ? eb.vectors : (vectors && vectors.length ? vectors : null)
            const rep = basisVectors && basisVectors.length ? basisVectors[0] : (vectors && vectors.length ? vectors[0] : null)

            // Determine original eigenvalue index for multiplicities.
            let matchedIndex = null
            if (ev && Array.isArray(ev) && ev.length === esList.length && ev[i]) {
              matchedIndex = i
            }
            try {
              if (matchedIndex === null && rep && flat.eigenvectors && Array.isArray(flat.eigenvectors.data)) {
                const cols = flat.eigenvectors.cols || (flat.eigenvectors.data[0] ? flat.eigenvectors.data[0].length : 0)
                for (let j = 0; j < cols; j++) {
                  const col = flat.eigenvectors.data.map(r => (r && r[j] !== undefined) ? r[j] : 0)
                  let dist = 0
                  for (let k = 0; k < col.length && k < rep.length; k++) dist += Math.pow((col[k] || 0) - (rep[k] || 0), 2)
                  if (Math.sqrt(dist) <= 1e-8) { matchedIndex = j; break }
                }
              }
            } catch {
              // ignore matching errors
            }

            // Fallback: if still not matched, try matching by eigenvalue numeric closeness
            if (matchedIndex === null && ev && Array.isArray(ev) && ev.length > 0 && eigen) {
              for (let j = 0; j < ev.length; j++) {
                const v = ev[j]
                if (!v || !eigen) continue
                const dx = (v.real || 0) - (eigen.real || 0)
                const dy = (v.imag || 0) - (eigen.imag || 0)
                if (Math.hypot(dx, dy) <= 1e-8) { matchedIndex = j; break }
              }
            }

            // If we found a matched original index, populate the eigen value for this entry
            if (matchedIndex !== null && ev && Array.isArray(ev) && ev[matchedIndex]) {
              eigen = ev[matchedIndex]
            }

            per.push({
              eigenvalue: eigen,
              algebraicMultiplicity: (alg && matchedIndex !== null && alg[matchedIndex] !== undefined) ? alg[matchedIndex] : null,
              geometricMultiplicity: (geom && matchedIndex !== null && geom[matchedIndex] !== undefined) ? geom[matchedIndex] : null,
              // canonical field: `dimension` refers to eigenspace dimension (geometric multiplicity)
              dimension: (geom && matchedIndex !== null && geom[matchedIndex] !== undefined) ? geom[matchedIndex] : (esList && esList[i] && esList[i].vectors ? esList[i].vectors.length : 0),
              eigenspace: { vectors: vectors, dimension: vectors ? vectors.length : 0 },
              eigenbasis: basisVectors ? { vectors: basisVectors, dimension: basisVectors.length } : null,
              representativeEigenvector: rep
            })
          }
          flat.eigenInformationPerValue = per
        }
    }
  } catch {
    // ignore fallback construction errors
  }
  // Backwards-compatible aliases for older frontend keys
  if (flat.invertible !== undefined && flat.invertibility === undefined) flat.invertibility = flat.invertible
  if (flat.defective !== undefined && flat.defectivity === undefined) flat.defectivity = flat.defective
  if (flat.diagonalizable !== undefined && flat.diagonalizability === undefined) flat.diagonalizability = flat.diagonalizable
  if (flat.persymmetric !== undefined && flat.persymetric === undefined) flat.persymetric = flat.persymmetric
  if (flat.singular !== undefined && flat.singularity === undefined) flat.singularity = flat.singular

  // Add spectrum location and stability labels derived from eigenvalues (no backend dependency).
  if (flat.spectrumLocation === undefined || flat.odeStability === undefined || flat.discreteStability === undefined) {
    const spectrumInfo = classifySpectrum(flat.eigenvalues || [])
    if (flat.spectrumLocation === undefined) flat.spectrumLocation = spectrumInfo.location
    if (flat.odeStability === undefined) flat.odeStability = spectrumInfo.odeStability
    if (flat.discreteStability === undefined) flat.discreteStability = spectrumInfo.discreteStability
  }

  // Derive GEMM performance (GFLOPs) for a 512x512 run from any benchmark payloads
  try {
    const benchCandidates = []
    if (resp && typeof resp === 'object') {
      if (resp.benchmark) benchCandidates.push(resp.benchmark)
      if (resp.benchmarks) benchCandidates.push(resp.benchmarks)
      if (resp.cpu && resp.cpu.benchmark) benchCandidates.push(resp.cpu.benchmark)
      if (resp.cpu && resp.cpu.benchmarks) benchCandidates.push(resp.cpu.benchmarks)
      if (resp._raw && resp._raw.benchmark) benchCandidates.push(resp._raw.benchmark)
    }

    let gemmGflops = null

    function inspectArray(arr) {
      if (!Array.isArray(arr)) return
      for (const item of arr) {
        if (!item || typeof item !== 'object') continue
        const name = (item.op || item.name || item.type || '').toString().toLowerCase()
        const n = Number(item.n || item.size || item.matrixSize || item.dim || 0) || 0
        // prefer explicit gflops/flopsPerSec if provided
        if (/gemm/.test(name) || /matmul|multiply/.test(name)) {
          if (item.gflops !== undefined && Number.isFinite(Number(item.gflops))) {
            gemmGflops = Number(item.gflops)
            return true
          }
          if (item.flopsPerSec !== undefined && Number.isFinite(Number(item.flopsPerSec))) {
            gemmGflops = Number(item.flopsPerSec) / 1e9
            return true
          }
          if (item.ms !== undefined && Number.isFinite(Number(item.ms))) {
            const size = n || 512
            const flops = 2 * Math.pow(size, 3)
            gemmGflops = (flops * 1000) / (Number(item.ms) * 1e9)
            return true
          }
        }
        // nested result arrays
        for (const v of Object.values(item)) {
          if (Array.isArray(v)) {
            if (inspectArray(v)) return true
          }
        }
      }
      return false
    }

    for (const c of benchCandidates) {
      if (!c) continue
      if (Array.isArray(c)) {
        if (inspectArray(c)) break
      } else if (typeof c === 'object') {
        // common shapes: { results: [...] } or { runs: [...] }
        if (inspectArray(c.results || c.runs || c.data || c)) break
      }
    }

    // If we didn't find a GEMM-specific measure but there's a generic average, use it as fallback
    if (gemmGflops === null) {
      const fallback = resp && resp.cpu && resp.cpu.benchmark && (resp.cpu.benchmark.gflops || resp.cpu.benchmark.gflopsAvg)
      if (fallback !== undefined && Number.isFinite(Number(fallback))) {
        gemmGflops = Number(fallback)
      }
    }

    if (gemmGflops !== null) {
      flat.cpu = flat.cpu || {}
      flat.cpu.benchmark = flat.cpu.benchmark || {}
      // expose under the existing key UI expects: `gflopsAvg` (value is in GFLOPs)
      flat.cpu.benchmark.gflopsAvg = gemmGflops
      // also expose a top-level cpu.gflops for direct consumption
      flat.cpu.gflops = gemmGflops
    }
  } catch {
    // ignore benchmark extraction errors
  }

  return flat
}

function parseComplexLike(value) {
  if (value === null || value === undefined) return { real: 0, imag: 0 }
  if (typeof value === 'number') return { real: value, imag: 0 }
  if (Array.isArray(value)) {
    const real = Number(value[0])
    const imag = Number(value[1])
    return { real: Number.isFinite(real) ? real : 0, imag: Number.isFinite(imag) ? imag : 0 }
  }
  if (typeof value === 'string') {
    const s = value.trim()
    if (s === 'i' || s === '+i') return { real: 0, imag: 1 }
    if (s === '-i') return { real: 0, imag: -1 }
    if (s.toLowerCase().includes('i')) {
      const core = s.replace(/i$/i, '')
      let splitPos = -1
      for (let i = core.length - 1; i > 0; i--) {
        const ch = core[i]
        if (ch === '+' || ch === '-') { splitPos = i; break }
      }
      if (splitPos === -1) {
        const imag = Number(core)
        return { real: 0, imag: Number.isFinite(imag) ? imag : 0 }
      }
      const real = Number(core.slice(0, splitPos))
      const imag = Number(core.slice(splitPos))
      return { real: Number.isFinite(real) ? real : 0, imag: Number.isFinite(imag) ? imag : 0 }
    }
    const real = Number(s)
    return { real: Number.isFinite(real) ? real : 0, imag: 0 }
  }
  if (typeof value === 'object') {
    const real = Number(value.real ?? value.r ?? 0)
    const imag = Number(value.imag ?? value.i ?? 0)
    return { real: Number.isFinite(real) ? real : 0, imag: Number.isFinite(imag) ? imag : 0 }
  }
  return { real: 0, imag: 0 }
}

function classifySpectrum(eigenvalues, tol = 1e-8) {
  if (!Array.isArray(eigenvalues) || eigenvalues.length === 0) {
    return { location: 'Unknown', odeStability: 'Unknown', discreteStability: 'Unknown' }
  }

  const parsed = eigenvalues.map(parseComplexLike)
  const realParts = parsed.map(v => v.real)
  const imagParts = parsed.map(v => v.imag)
  const mags = parsed.map(v => Math.hypot(v.real, v.imag))

  const allReal = imagParts.every(im => Math.abs(im) <= tol)
  const allRealNeg = realParts.every(re => re < -tol)
  const allRealPos = realParts.every(re => re > tol)
  const allRealNonPos = realParts.every(re => re <= tol)
  const allRealNonNeg = realParts.every(re => re >= -tol)
  const allImagAxis = realParts.every(re => Math.abs(re) <= tol)
  const anyRealPos = realParts.some(re => re > tol)

  let location = 'Mixed/General (across half-planes)'
  if (allImagAxis) {
    location = 'Imaginary Axis (Re λ = 0)'
  } else if (allRealNeg) {
    location = 'Left Half-Plane (Hurwitz)'
  } else if (allRealPos) {
    location = 'Right Half-Plane'
  } else if (allRealNonPos) {
    location = 'Closed Left Half-Plane (Re λ ≤ 0)'
  } else if (allRealNonNeg) {
    location = 'Closed Right Half-Plane (Re λ ≥ 0)'
  } else if (allReal) {
    location = 'Real Axis (mixed sign)'
  }

  let odeStability = 'Unstable'
  if (allRealNeg) {
    odeStability = 'Asymptotically Stable (Hurwitz)'
  } else if (!anyRealPos && allRealNonPos) {
    odeStability = 'Marginal (Re λ ≤ 0)'
  }

  const allInside = mags.every(m => m < 1 - tol)
  const anyOutside = mags.some(m => m > 1 + tol)
  const allInsideOrOn = mags.every(m => m <= 1 + tol)

  let discreteStability = 'Unstable (outside unit disk)'
  if (allInside) {
    discreteStability = 'Asymptotically Stable (inside unit disk)'
  } else if (!anyOutside && allInsideOrOn) {
    discreteStability = 'Marginal (on unit circle)'
  }

  return { location, odeStability, discreteStability }
}

