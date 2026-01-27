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
  } catch (err) {
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
  } catch (err) {
    return null
  }
}

/**
 * Persist diagnostics to session storage.
 *
 * @param {string} matrixString
 * @param {any} diagnostics
 */
export function cacheDiagnostics(matrixString, diagnostics) {
  if (!matrixString || !diagnostics) return
  try {
    sessionStorage.setItem(
      cacheKey(matrixString),
      JSON.stringify({
        ts: Date.now(),
        data: diagnostics
      })
    )
  } catch (err) {
    // ignore cache failures
  }
}

/**
 * Fetch diagnostics from the backend API.
 *
 * @param {number[][]} matrixData
 * @returns {Promise<any>}
 */
export async function fetchDiagnostics(matrixData) {
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
            } catch (e) {
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
  } catch (e) {
    // ignore fallback construction errors
  }
  // Backwards-compatible aliases for older frontend keys
  if (flat.invertible !== undefined && flat.invertibility === undefined) flat.invertibility = flat.invertible
  if (flat.defective !== undefined && flat.defectivity === undefined) flat.defectivity = flat.defective
  if (flat.diagonalizable !== undefined && flat.diagonalizability === undefined) flat.diagonalizability = flat.diagonalizable
  if (flat.persymmetric !== undefined && flat.persymetric === undefined) flat.persymetric = flat.persymmetric
  if (flat.singular !== undefined && flat.singularity === undefined) flat.singularity = flat.singular

  return flat
}

