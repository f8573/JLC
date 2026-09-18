/**
 * Strip spaces, tabs, and newlines so matrix files can be parsed uniformly.
 *
 * @param {string} text
 * @returns {string}
 */
export function sanitizeMatrixText(text) {
  return (text ?? '').toString().replace(/[ \t\r\n]+/g, '')
}

const NUMBER_PATTERN = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/

function ensureRectangular(matrix, label) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new Error(`${label} must be a non-empty array of arrays`)
  }
  let cols = null
  matrix.forEach((row, rowIdx) => {
    if (!Array.isArray(row) || row.length === 0) {
      throw new Error(`${label} row ${rowIdx + 1} must be a non-empty array`)
    }
    if (cols === null) {
      cols = row.length
    } else if (row.length !== cols) {
      throw new Error(`${label} is malformed: every row must have exactly ${cols} columns`)
    }
  })
  return { rows: matrix.length, cols: cols ?? 0 }
}

function parseStrictNumber(value, context) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new Error(`Invalid numeric value at ${context}`)
    }
    return value
  }
  if (typeof value === 'string' && NUMBER_PATTERN.test(value)) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  throw new Error(`Invalid numeric value at ${context}`)
}

function parseRealMatrix(matrix, label) {
  ensureRectangular(matrix, label)
  return matrix.map((row, rowIdx) =>
    row.map((value, colIdx) => parseStrictNumber(value, `${label}[${rowIdx}][${colIdx}]`))
  )
}

function splitRealImag(core) {
  let splitPos = -1
  for (let i = 1; i < core.length; i++) {
    const ch = core[i]
    const prev = core[i - 1]
    if ((ch === '+' || ch === '-') && prev !== 'e' && prev !== 'E') {
      splitPos = i
    }
  }
  if (splitPos === -1) return [null, core]
  return [core.slice(0, splitPos), core.slice(splitPos)]
}

function parseComplexToken(value, context) {
  if (typeof value === 'number') {
    return { real: parseStrictNumber(value, context), imag: 0 }
  }
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Invalid complex value at ${context}`)
  }

  const token = value.trim()
  const lower = token.toLowerCase()
  const iCount = (lower.match(/i/g) || []).length

  if (iCount === 0) {
    return { real: parseStrictNumber(token, context), imag: 0 }
  }
  if (iCount !== 1 || !lower.endsWith('i')) {
    throw new Error(`Invalid complex value at ${context}`)
  }

  const core = token.slice(0, -1)
  if (core === '' || core === '+') return { real: 0, imag: 1 }
  if (core === '-') return { real: 0, imag: -1 }

  const [realPart, imagPart] = splitRealImag(core)
  if (realPart === null) {
    return { real: 0, imag: parseStrictNumber(imagPart, context) }
  }

  const imag =
    imagPart === '+' ? 1 :
    imagPart === '-' ? -1 :
    parseStrictNumber(imagPart, context)

  return {
    real: parseStrictNumber(realPart, context),
    imag
  }
}

function isComplexLike(value) {
  if (Array.isArray(value)) return value.length === 2
  if (value && typeof value === 'object') {
    return value.real !== undefined ||
      value.imag !== undefined ||
      value.r !== undefined ||
      value.i !== undefined
  }
  return typeof value === 'string' && value.toLowerCase().includes('i')
}

function parseComplexCell(value, context) {
  if (typeof value === 'number') {
    return { real: parseStrictNumber(value, context), imag: 0 }
  }
  if (typeof value === 'string') {
    return parseComplexToken(value, context)
  }
  if (Array.isArray(value)) {
    if (value.length !== 2) {
      throw new Error(`Invalid complex value at ${context}`)
    }
    return {
      real: parseStrictNumber(value[0], context),
      imag: parseStrictNumber(value[1], context)
    }
  }
  if (value && typeof value === 'object') {
    const real = value.real ?? value.r ?? 0
    const imag = value.imag ?? value.i ?? 0
    return {
      real: parseStrictNumber(real, context),
      imag: parseStrictNumber(imag, context)
    }
  }
  throw new Error(`Invalid complex value at ${context}`)
}

function normalizeArrayMatrix(matrix, label) {
  ensureRectangular(matrix, label)
  const hasComplexEntries = matrix.some((row) => row.some(isComplexLike))
  if (!hasComplexEntries) {
    return parseRealMatrix(matrix, label)
  }

  const real = []
  const imag = []
  matrix.forEach((row, rowIdx) => {
    const realRow = []
    const imagRow = []
    row.forEach((value, colIdx) => {
      const parsed = parseComplexCell(value, `${label}[${rowIdx}][${colIdx}]`)
      realRow.push(parsed.real)
      imagRow.push(parsed.imag)
    })
    real.push(realRow)
    imag.push(imagRow)
  })

  return { data: real, imag }
}

function normalizeComplexObjectMatrix(obj, label) {
  if (!Array.isArray(obj.data)) {
    throw new Error(`${label}.data must be a matrix`)
  }
  const data = parseRealMatrix(obj.data, `${label}.data`)
  if (obj.imag === undefined || obj.imag === null) {
    return data
  }
  if (!Array.isArray(obj.imag)) {
    throw new Error(`${label}.imag must be a matrix`)
  }
  const imag = parseRealMatrix(obj.imag, `${label}.imag`)
  const a = ensureRectangular(data, `${label}.data`)
  const b = ensureRectangular(imag, `${label}.imag`)
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(`${label} is malformed: real and imaginary matrices must have identical dimensions`)
  }
  return { data, imag }
}

/**
 * Normalize parsed matrix JSON into either:
 * - number[][]
 * - { data: number[][], imag: number[][] }
 *
 * @param {any} parsed
 * @returns {number[][] | {data: number[][], imag: number[][]}}
 */
export function normalizeParsedMatrixData(parsed) {
  if (Array.isArray(parsed)) {
    return normalizeArrayMatrix(parsed, 'matrix')
  }
  if (parsed && typeof parsed === 'object') {
    return normalizeComplexObjectMatrix(parsed, 'matrix')
  }
  throw new Error('Matrix payload must be an array of arrays')
}

function quoteInlineMatrixTokens(input) {
  let out = ''
  let token = ''
  const flush = () => {
    if (!token) return
    out += `"${token}"`
    token = ''
  }

  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (ch === '[' || ch === ']' || ch === ',') {
      flush()
      out += ch
    } else {
      token += ch
    }
  }
  flush()
  return out
}

function parseInlineComplexMatrix(input) {
  let parsed
  try {
    parsed = JSON.parse(quoteInlineMatrixTokens(input))
  } catch {
    throw new Error('Malformed complex matrix format')
  }
  ensureRectangular(parsed, 'matrix')

  const real = []
  const imag = []
  parsed.forEach((row, rowIdx) => {
    const realRow = []
    const imagRow = []
    row.forEach((token, colIdx) => {
      const value = parseComplexToken(token, `matrix[${rowIdx}][${colIdx}]`)
      realRow.push(value.real)
      imagRow.push(value.imag)
    })
    real.push(realRow)
    imag.push(imagRow)
  })
  return { data: real, imag }
}

/**
 * Parse matrix text from freeform user input or uploaded file.
 *
 * Supported formats:
 * - [[1,2],[3,4]]
 * - [[1,2],[3,4]];[[5,6],[7,8]]
 * - [[1+5i,2+6i],[3+7i,4+8i]]
 *
 * @param {string} rawText
 * @returns {number[][] | {data: number[][], imag: number[][]}}
 */
export function parseMatrixInputText(rawText) {
  const text = sanitizeMatrixText(rawText)
  if (!text) {
    throw new Error('Uploaded file is empty')
  }

  if (text.includes(';')) {
    const parts = text.split(';')
    if (parts.length !== 2 || !parts[0] || !parts[1]) {
      throw new Error('Complex split format must be realMatrix;imagMatrix')
    }

    let realRaw
    let imagRaw
    try {
      realRaw = JSON.parse(parts[0])
      imagRaw = JSON.parse(parts[1])
    } catch {
      throw new Error('Complex split format must contain valid JSON arrays')
    }

    const real = parseRealMatrix(realRaw, 'real matrix')
    const imag = parseRealMatrix(imagRaw, 'imaginary matrix')
    const a = ensureRectangular(real, 'real matrix')
    const b = ensureRectangular(imag, 'imaginary matrix')
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error('Real and imaginary matrices must have the same dimensions')
    }
    return { data: real, imag }
  }

  if (text.toLowerCase().includes('i')) {
    return parseInlineComplexMatrix(text)
  }

  let parsed
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('Matrix file must contain valid JSON matrix data')
  }
  return normalizeParsedMatrixData(parsed)
}

/**
 * Convert matrix payload into real-valued data for the diagnostics API.
 * Complex entries are reduced to their real part.
 *
 * @param {any} matrixData
 * @returns {number[][]}
 */
export function matrixToRealData(matrixData) {
  const normalized = normalizeParsedMatrixData(matrixData)
  if (Array.isArray(normalized)) {
    return normalized
  }
  return parseRealMatrix(normalized.data, 'matrix.data')
}
