#!/usr/bin/env node
// tools/concurrent_benchmark.js
// Issue many concurrent GET requests to the backend benchmark endpoint
// and poll /api/status to observe queuedJobs and status transitions.

const axios = require('axios');
const minimist = require('minimist');

const argv = minimist(process.argv.slice(2), {
  string: ['url'],
  integer: ['concurrency', 'total', 'pollInterval'],
  boolean: ['pollStatus'],
  default: {
    url: 'http://localhost:8080/api/benchmark/diagnostic?sizex=512&sizey=512&test=GEMM&iterations=1',
    concurrency: 20,
    total: 100,
    pollStatus: true,
    pollInterval: 1000
  }
});

const TARGET = argv.url;
const CONCURRENCY = Number(argv.concurrency);
const TOTAL = Number(argv.total);
const POLL_STATUS = !!argv.pollStatus;
const POLL_INTERVAL = Number(argv.pollInterval);

console.log(`Concurrent benchmark: target=${TARGET} concurrency=${CONCURRENCY} total=${TOTAL}`);

let inFlight = 0;
let sent = 0;
let succeeded = 0;
let failed = 0;

async function doRequest(i) {
  inFlight++;
  try {
    const t0 = Date.now();
    const resp = await axios.get(TARGET, { timeout: 120000 });
    const dt = Date.now() - t0;
    succeeded++;
    console.log(`[${i}] status=${resp.status} time=${dt}ms inFlight=${inFlight}`);
  } catch (err) {
    failed++;
    console.log(`[${i}] ERROR ${err.message} inFlight=${inFlight}`);
  } finally {
    inFlight--;
  }
}

async function runAll() {
  const tasks = [];
  let nextIndex = 1;

  // status poller
  let poller = null;
  if (POLL_STATUS) {
    poller = setInterval(async () => {
      try {
        const s = await axios.get('http://localhost:8080/api/status', { timeout: 5000 });
        const st = s.data && s.data.status ? s.data.status : JSON.stringify(s.data);
        const q = s.data && s.data.cpu && s.data.cpu.queuedJobs != null ? s.data.cpu.queuedJobs : '-';
        console.log(`STATUS: ${st} queuedJobs=${q} (inFlight=${inFlight} sent=${sent} succ=${succeeded} fail=${failed})`);
      } catch (e) {
        console.log('STATUS: error', e.message);
      }
    }, POLL_INTERVAL);
  }

  while (nextIndex <= TOTAL) {
    if (inFlight < CONCURRENCY) {
      const idx = nextIndex++;
      sent++;
      // start without awaiting
      doRequest(idx);
    } else {
      // wait a bit
      await new Promise(res => setTimeout(res, 50));
    }
  }

  // wait for inflight to drain
  while (inFlight > 0) {
    await new Promise(res => setTimeout(res, 200));
  }

  if (poller) clearInterval(poller);
  console.log('Run complete:', { sent, succeeded, failed });
}

runAll().catch(err => {
  console.error('Fatal', err);
  process.exit(2);
});
