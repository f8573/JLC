// frontend/tests/browser-worker.js
// Worker script for browser-stress.html
// Receives config via postMessage and runs loop sending matrices to target endpoint.
// Messages to main: {type:'stats', sent, pendingDelta, failed, latency, queueHint}, {type:'log', text}

self.onmessage = async function(e) {
  const msg = e.data;
  if (msg.cmd === 'start') {
    runWorker(msg).catch(err => {
      postMessage({ type:'log', text: 'worker error: ' + (err.stack || err) });
    });
  } else if (msg.cmd === 'stop') {
    self.stopRequested = true;
  }
};

async function sendJson(url, rows, cols, floatBuf) {
  // Convert flat float buffer to nested array expected by backend { matrix: [[...],[...]] }
  const f32 = new Float32Array(floatBuf);
  const matrix = new Array(rows);
  for (let r = 0; r < rows; ++r) {
    const row = new Array(cols);
    const base = r * cols;
    for (let c = 0; c < cols; ++c) row[c] = f32[base + c];
    matrix[r] = row;
  }
  const body = JSON.stringify({ matrix });
  const start = performance.now();
  const resp = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
  const latency = performance.now() - start;
  let queueHint = null;
  try {
    const ct = resp.headers.get('content-type') || '';
    if (resp.headers.get('x-queue-length')) queueHint = resp.headers.get('x-queue-length');
    else if (ct.includes('application/json')) {
      const bodyJson = await resp.json();
      if (bodyJson && (bodyJson.queue_length || bodyJson.queue)) queueHint = bodyJson.queue_length || bodyJson.queue;
    }
  } catch (e) { /* ignore */ }
  return { status: resp.status, latency, queueHint };
}

async function sendBinary(url, rows, cols, headerBuf, floatBuf) {
  // Keep binary path for compatibility but server currently expects JSON; use only if backend supports it.
  const blob = new Blob([headerBuf, floatBuf], { type: 'application/octet-stream' });
  const start = performance.now();
  const resp = await fetch(url, { method: 'POST', headers: { 'X-Matrix-Encoding': 'binary-float32' }, body: blob });
  const latency = performance.now() - start;
  let queueHint = null;
  try {
    if (resp.headers.get('x-queue-length')) queueHint = resp.headers.get('x-queue-length');
    else {
      const ct = resp.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const bodyJson = await resp.json();
        if (bodyJson && (bodyJson.queue_length || bodyJson.queue)) queueHint = bodyJson.queue_length || bodyJson.queue;
      }
    }
  } catch (e) {}
  return { status: resp.status, latency, queueHint };
}

async function runWorker(cfg) {
  const { url, encoding, matrix, mode, rps, duration, burstCount, pattern, clientId, totalClients } = cfg;
  const rows = matrix|0;
  const cols = matrix|0;
  self.stopRequested = false;

  // Pre-allocate buffers for this worker to reuse
  const HEADER_BYTES = 12;
  const MATRIX_ELEMS = rows * cols;
  const MATRIX_BYTES = MATRIX_ELEMS * 4;
  const totalBytes = HEADER_BYTES + MATRIX_BYTES;

  // create combined buffer for binary; for JSON we'll reuse float area portion
  const abuffer = new ArrayBuffer(totalBytes);
  const headerView = new DataView(abuffer, 0, HEADER_BYTES);
  headerView.setUint8(0, 'M'.charCodeAt(0));
  headerView.setUint8(1, 'T'.charCodeAt(0));
  headerView.setUint8(2, 'X'.charCodeAt(0));
  headerView.setUint8(3, '1'.charCodeAt(0));
  headerView.setUint32(4, rows, true);
  headerView.setUint32(8, cols, true);
  const floatBuf = new Float32Array(abuffer, HEADER_BYTES, MATRIX_ELEMS);

  function fill(pattern) {
    if (pattern === 'zeros') {
      floatBuf.fill(0);
    } else if (pattern === 'ones') {
      floatBuf.fill(1);
    } else {
      let seed = (clientId + 1) * 1234567;
      for (let i = 0; i < MATRIX_ELEMS; ++i) {
        seed = (1664525 * seed + 1013904223) >>> 0;
        floatBuf[i] = (seed >>> 0) / 4294967296;
      }
    }
  }

  const startTime = performance.now();
  const endTime = startTime + duration * 1000;

  if (mode === 'burst') {
    postMessage({ type:'log', text: `Worker ${clientId} sending ${burstCount} requests (burst)`});
    for (let i = 0; i < burstCount && !self.stopRequested; ++i) {
      fill(pattern);
      const payloadFloatBuf = floatBuf.buffer.slice(floatBuf.byteOffset, floatBuf.byteOffset + floatBuf.byteLength);
      try {
        let res;
        if (encoding === 'binary') {
          res = await sendBinary(url, rows, cols, abuffer.slice(0, HEADER_BYTES), payloadFloatBuf);
        } else {
          res = await sendJson(url, rows, cols, payloadFloatBuf);
        }
        postMessage({ type: 'stats', sent: 1, pendingDelta: 0, failed: res.status >= 400 ? 1 : 0, latency: res.latency, queueHint: res.queueHint });
      } catch (err) {
        postMessage({ type: 'stats', sent: 1, pendingDelta: 0, failed: 1 });
      }
    }
    postMessage({ type:'log', text: `Worker ${clientId} burst complete` });
    return;
  }

  const perClientRps = Math.max(1, Math.floor(rps / totalClients));
  const delayMs = 1000 / perClientRps;

  postMessage({ type:'log', text: `Worker ${clientId} sustained mode: ~${perClientRps} rps (delay ${delayMs.toFixed(1)}ms)` });

  while (!self.stopRequested && performance.now() < endTime) {
    const t0 = performance.now();
    fill(pattern);
    const payloadFloatBuf = floatBuf.buffer.slice(floatBuf.byteOffset, floatBuf.byteOffset + floatBuf.byteLength);
    try {
      if (encoding === 'binary') {
        const res = await sendBinary(url, rows, cols, abuffer.slice(0, HEADER_BYTES), payloadFloatBuf);
        postMessage({ type: 'stats', sent: 1, pendingDelta: 0, failed: res.status >= 400 ? 1 : 0, latency: res.latency, queueHint: res.queueHint });
      } else {
        const res = await sendJson(url, rows, cols, payloadFloatBuf);
        postMessage({ type: 'stats', sent: 1, pendingDelta: 0, failed: res.status >= 400 ? 1 : 0, latency: res.latency, queueHint: res.queueHint });
      }
    } catch (err) {
      postMessage({ type: 'stats', sent: 1, pendingDelta: 0, failed: 1 });
    }
    const t1 = performance.now();
    const elapsed = t1 - t0;
    const sleep = Math.max(0, delayMs - elapsed);
    if (sleep > 0) {
      await new Promise(res => setTimeout(res, sleep));
    }
  }
  postMessage({ type:'log', text: `Worker ${clientId} finished` });
}
