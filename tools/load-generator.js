#!/usr/bin/env node
/**
 * tools/load-generator.js
 *
 * Node.js load generator for submitting 512x512 (configurable) float32 matrices
 * to a target endpoint. Supports JSON (base64 float32) and binary payloads
 * with a small header. Designed to reuse buffers via a pool to avoid OOM.
 *
 * Usage examples:
 *  node tools/load-generator.js --url http://localhost:8080/api/diagnostics --concurrency 200 --rps 500 --duration 60 --encoding json
 *
 * Dependencies: axios, minimist
 *
 * Key features:
 * - concurrency limit
 * - request rate (requests per second)
 * - burst mode and ramping (see CLI options)
 * - binary and JSON payload encodings
 * - buffer pool to reuse memory
 * - latency, throughput, error logging to console and CSV
 * - optional Prometheus Pushgateway support
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const minimist = require('minimist');

const argv = minimist(process.argv.slice(2), {
  string: ['url', 'encoding', 'output', 'prometheus'],
  integer: ['concurrency', 'rps', 'duration', 'matrixSize', 'timeout', 'retries', 'poolSize', 'burstSize', 'rampStep', 'rampInterval'],
  boolean: ['burst'],
  alias: {
    u: 'url',
    c: 'concurrency',
    r: 'rps',
    d: 'duration',
    e: 'encoding',
    o: 'output',
    p: 'prometheus'
  },
  default: {
    url: 'http://localhost:8080/api/diagnostics',
    concurrency: 100,
    rps: 200,
    duration: 60, // seconds
    encoding: 'json', // 'json' or 'binary'
    matrixSize: 512,
    timeout: 60_000,
    retries: 2,
    poolSize: 50,
    burst: false,
    burstSize: 1000,
    rampStep: 50,
    rampInterval: 10 // seconds
  }
});

// Config
const TARGET_URL = argv.url;
const CONCURRENCY = Number(argv.concurrency);
const RPS = Number(argv.rps);
const DURATION = Number(argv.duration); // seconds
const ENCODING = argv.encoding === 'binary' ? 'binary' : 'json';
const MATRIX_N = Number(argv.matrixSize);
const TIMEOUT = Number(argv.timeout);
const RETRIES = Number(argv.retries);
const POOL_SIZE = Math.max(1, Number(argv.poolSize));
const OUTPUT_CSV = argv.output || null;
const PROM_PUSHGATEWAY = argv.prometheus || null;
const BURST = !!argv.burst;
const BURST_SIZE = Number(argv.burstSize);
const RAMP_STEP = Number(argv.rampStep);
const RAMP_INTERVAL = Number(argv.rampInterval);

if (!TARGET_URL) {
  console.error('Please provide --url');
  process.exit(1);
}

// Derived
const MATRIX_ELEMS = MATRIX_N * MATRIX_N;
const MATRIX_BYTES = MATRIX_ELEMS * 4; // float32
const HEADER_BYTES = 12; // for binary header in our format
console.log(`Load generator config:
 - target: ${TARGET_URL}
 - concurrency: ${CONCURRENCY}
 - rps: ${RPS} (if burst mode, this is ignored)
 - duration: ${DURATION}s
 - encoding: ${ENCODING}
 - matrix: ${MATRIX_N}x${MATRIX_N} (~${(MATRIX_BYTES/1024/1024).toFixed(2)} MB per matrix)
 - timeout: ${TIMEOUT}ms
 - retries: ${RETRIES}
 - pool size: ${POOL_SIZE}
 - burst mode: ${BURST ? 'ON (burstSize='+BURST_SIZE+')' : 'off'}
 - ramp step: ${RAMP_STEP} every ${RAMP_INTERVAL}s
`);

//
// Basic CSV logging
//
let csvStream = null;
if (OUTPUT_CSV) {
  const outdir = path.dirname(OUTPUT_CSV);
  if (!fs.existsSync(outdir)) fs.mkdirSync(outdir, { recursive: true });
  const existed = fs.existsSync(OUTPUT_CSV);
  csvStream = fs.createWriteStream(OUTPUT_CSV, { flags: 'a' });
  if (!existed) {
    csvStream.write('timestamp_ms,latency_ms,http_status,result,queue_hint,attempt\n');
  }
}

function csvLog(row) {
  if (!csvStream) return;
  const line = [
    row.ts || Date.now(),
    row.latency == null ? '' : Math.round(row.latency),
    row.status || '',
    row.result || '',
    row.queueHint == null ? '' : row.queueHint,
    row.attempt || 0
  ].join(',') + '\n';
  csvStream.write(line);
}

//
// Buffer pool: create POOL_SIZE buffers for payloads to reuse
//
const pool = [];
for (let i = 0; i < POOL_SIZE; ++i) {
  // For binary encoding we'll write header+float data into an allocated buffer
  const buf = Buffer.allocUnsafe(HEADER_BYTES + MATRIX_BYTES);
  // create Float32Array view over the payload area (after header)
  const floatView = new Float32Array(buf.buffer, buf.byteOffset + HEADER_BYTES, MATRIX_ELEMS);
  pool.push({ buf, floatView, inUse: false });
}
function acquireBuffer() {
  for (const p of pool) {
    if (!p.inUse) {
      p.inUse = true;
      return p;
    }
  }
  // none available; allocate a temp buffer (avoid OOM by limiting concurrency)
  const buf = Buffer.allocUnsafe(HEADER_BYTES + MATRIX_BYTES);
  const floatView = new Float32Array(buf.buffer, buf.byteOffset + HEADER_BYTES, MATRIX_ELEMS);
  return { buf, floatView, inUse: true, temporary: true };
}
function releaseBuffer(p) {
  p.inUse = false;
  if (p.temporary) {
    // free: let GC collect
  }
}

//
// Matrix fill: fill with simple pattern (random by default)
//
function fillMatrix(floatView, pattern = 'random') {
  if (pattern === 'zeros') {
    floatView.fill(0);
    return;
  }
  if (pattern === 'ones') {
    floatView.fill(1);
    return;
  }
  // random: we use a simple xorshift PRNG for speed and deterministic behaviour
  let seed = 123456789;
  function rand() {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    // convert to [0,1)
    return (seed >>> 0) / 4294967296;
  }
  const n = floatView.length;
  for (let i = 0; i < n; ++i) {
    floatView[i] = rand();
  }
}

//
// Binary format header: 12 bytes
// 0-3: ASCII 'MTX1' magic
// 4-7: uint32 rows (LE)
// 8-11: uint32 cols (LE)
// follow by float32 payload (row-major)
//
function writeBinaryHeader(buffer, rows, cols) {
  buffer.write('MTX1', 0, 4, 'ascii');
  buffer.writeUInt32LE(rows, 4);
  buffer.writeUInt32LE(cols, 8);
}

//
// Scheduler & sending logic
//
let activeRequests = 0;
let totalSent = 0;
let totalSucceeded = 0;
let totalFailed = 0;
let latencies = []; // ms
let startTime = Date.now();
let stopRequested = false;
let perMinuteStats = {}; // minute timestamp -> {sent, success, failed, avgLatency}

async function sendOne(payloadBuf, isBinary, attempt = 0) {
  const start = Date.now();
  try {
    const headers = {};
    if (isBinary) {
      headers['Content-Type'] = 'application/octet-stream';
      headers['X-Matrix-Encoding'] = 'binary-float32';
    } else {
      headers['Content-Type'] = 'application/json';
      headers['X-Matrix-Encoding'] = 'base64-float32-json';
    }

    const axiosConfig = {
      url: TARGET_URL,
      method: 'POST',
      headers,
      timeout: TIMEOUT,
      data: payloadBuf,
      validateStatus: null // handle all statuses ourselves
    };

    const resp = await axios(axiosConfig);
    const latency = Date.now() - start;
    latencies.push(latency);

    // parse possible queue hints (header or body)
    let queueHint = null;
    if (resp.headers['x-queue-length']) {
      queueHint = resp.headers['x-queue-length'];
    } else if (resp.data && typeof resp.data === 'object' && (resp.data.queue_length || resp.data.queue)) {
      queueHint = resp.data.queue_length || resp.data.queue;
    }

    csvLog({ ts: Date.now(), latency, status: resp.status, result: resp.status >= 200 && resp.status < 300 ? 'ok' : 'error', queueHint, attempt });

    // interpret results
    if (resp.status >= 200 && resp.status < 300) {
      totalSucceeded++;
      return { ok: true, status: resp.status, queueHint, body: resp.data };
    } else if ((resp.status === 503 || resp.status === 429) && attempt < RETRIES) {
      // retry with backoff
      const backoffMs = Math.pow(2, attempt) * 250 + Math.random() * 100;
      await new Promise(res => setTimeout(res, backoffMs));
      return sendOne(payloadBuf, isBinary, attempt + 1);
    } else {
      totalFailed++;
      return { ok: false, status: resp.status, queueHint, body: resp.data };
    }

  } catch (err) {
    const latency = Date.now() - start;
    latencies.push(latency);
    csvLog({ ts: Date.now(), latency, status: 'ERR', result: 'exception', queueHint: '', attempt });
    if (attempt < RETRIES) {
      const backoffMs = Math.pow(2, attempt) * 250 + Math.random() * 100;
      await new Promise(res => setTimeout(res, backoffMs));
      return sendOne(payloadBuf, isBinary, attempt + 1);
    } else {
      totalFailed++;
      return { ok: false, status: 'ERR', error: err.message || String(err) };
    }
  } finally {
    // nothing
  }
}

async function createPayloadAndSend(isBinary, pattern = 'random') {
  // acquire buffer from pool
  const p = acquireBuffer();
  try {
    if (isBinary) {
      // header + payload in p.buf, we fill payload area using p.floatView
      fillMatrix(p.floatView, pattern);
      writeBinaryHeader(p.buf, MATRIX_N, MATRIX_N);
      // send the buffer slice that contains header+payload
      const slice = p.buf; // full buffer
      const res = await sendOne(slice, true);
      return res;
    } else {
      // for JSON encoding we avoid extra allocation by base64-ing the Buffer view of payload bytes
      // we will fill the payload portion starting at offset HEADER_BYTES but we only need payload bytes (no header)
      // Use a smaller temporary Buffer for payload portion to keep header bytes out of base64
      // For memory efficiency, create a Buffer view on the underlying buffer's payload portion
      // Node Buffer slice shares memory; use Buffer.from(p.buf.buffer, p.buf.byteOffset + HEADER_BYTES, MATRIX_BYTES)
        // fill via floatView (floatView already points to payload region)
        fillMatrix(p.floatView, pattern);
        // Build nested array (rows) expected by backend: { matrix: [[r0c0, r0c1,...],[r1c0,...], ...] }
        const rows = new Array(MATRIX_N);
        for (let r = 0; r < MATRIX_N; ++r) {
          const rowArr = new Array(MATRIX_N);
          const base = r * MATRIX_N;
          for (let c = 0; c < MATRIX_N; ++c) {
            rowArr[c] = p.floatView[base + c];
          }
          rows[r] = rowArr;
        }
        const json = { matrix: rows };
        const res = await sendOne(json, false);
      return res;
    }
  } finally {
    releaseBuffer(p);
  }
}

//
// Rate controller: token bucket
//
class TokenBucket {
  constructor(tokensPerSecond) {
    this.capacity = tokensPerSecond;
    this.tokens = tokensPerSecond;
    this.fillRate = tokensPerSecond;
    this.last = Date.now();
  }
  take(tokens = 1) {
    const now = Date.now();
    const delta = (now - this.last) / 1000;
    this.last = now;
    this.tokens = Math.min(this.capacity, this.tokens + delta * this.fillRate);
    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }
    return false;
  }
}

async function runLoad() {
  startTime = Date.now();
  const endTime = startTime + DURATION * 1000;
  let bucket = new TokenBucket(RPS);
  let scheduled = 0;
  let minuteWindow = Math.floor((Date.now() - startTime) / 60000);

  // Ramp logic: increase concurrency gradually if RAMP_STEP > 0
  let currentConcurrencyLimit = CONCURRENCY;
  let nextRampTime = Date.now() + RAMP_INTERVAL * 1000;

  if (BURST) {
    console.log(`Burst mode: sending ${BURST_SIZE} requests as fast as concurrency allows.`);
    const burstPromises = [];
    for (let i = 0; i < BURST_SIZE; ++i) {
      while (activeRequests >= currentConcurrencyLimit) {
        await new Promise(res => setTimeout(res, 5));
      }
      activeRequests++;
      totalSent++;
      const p = createPayloadAndSend(ENCODING === 'binary');
      p.finally(() => { activeRequests--; });
      burstPromises.push(p);
    }
    await Promise.all(burstPromises);
    console.log('Burst complete.');
    return;
  }

  while (!stopRequested && Date.now() < endTime) {
    // handle ramp
    if (Date.now() >= nextRampTime) {
      currentConcurrencyLimit = Math.min(20000, currentConcurrencyLimit + RAMP_STEP);
      nextRampTime += RAMP_INTERVAL * 1000;
      console.log(`Ramped concurrency limit to ${currentConcurrencyLimit}`);
    }
    // schedule up to concurrency and RPS
    if (activeRequests < currentConcurrencyLimit && bucket.take(1)) {
      activeRequests++;
      totalSent++;
      // schedule request in background
      (async () => {
        try {
          await createPayloadAndSend(ENCODING === 'binary');
        } catch (e) {
          // already logged
        } finally {
          activeRequests--;
        }
      })();
    } else {
      // wait briefly
      await new Promise(res => setTimeout(res, 5));
    }

    // per-minute stats
    const nowMinute = Math.floor((Date.now() - startTime) / 60000);
    if (nowMinute !== minuteWindow) {
      // print summary of last minute
      const all = latencies.splice(0); // empty latencies into arr
      const avg = all.length ? (all.reduce((a,b)=>a+b,0)/all.length) : 0;
      console.log(`[minute ${minuteWindow}] sent: ${totalSent}, success: ${totalSucceeded}, failed: ${totalFailed}, avgLatency(ms): ${avg.toFixed(1)}`);
      minuteWindow = nowMinute;
    }
  }

  // Wait for inflight to finish up to timeout
  const waitStart = Date.now();
  while (activeRequests > 0 && Date.now() - waitStart < TIMEOUT) {
    console.log(`Waiting for ${activeRequests} inflight requests to finish...`);
    await new Promise(res => setTimeout(res, 500));
  }

  // Final summary
  const durations = latencies;
  const avgLatency = durations.length ? (durations.reduce((a,b)=>a+b,0)/durations.length) : 0;
  console.log('Run complete.');
  console.log(`Total sent: ${totalSent}`);
  console.log(`Total succeeded: ${totalSucceeded}`);
  console.log(`Total failed: ${totalFailed}`);
  console.log(`Avg latency (ms): ${avgLatency.toFixed(1)}`);
}

runLoad().catch(err => {
  console.error('Fatal error in load run:', err);
  process.exit(2);
});
