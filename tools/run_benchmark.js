#!/usr/bin/env node
// Call the backend benchmark endpoint to (re)run the CPU benchmark.
const DEFAULT_URL = 'http://localhost:8080/api/benchmark/diagnostic?sizex=512&sizey=512&test=GEMM&iterations=3'

(async function(){
  const url = process.argv[2] || DEFAULT_URL
  console.log('Calling benchmark endpoint:', url)
  try {
    if (typeof fetch !== 'function') {
      console.error('Node fetch not available; use Node 18+')
      process.exit(2)
    }
    const res = await fetch(url)
    const txt = await res.text()
    console.log('Status:', res.status)
    console.log('Response:', txt.substring(0, 800))
  } catch (e) {
    console.error('Error calling benchmark:', e.message)
    process.exit(1)
  }
})()
