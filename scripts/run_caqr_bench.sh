#!/usr/bin/env bash
set -euo pipefail

THREADS=${1:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GRADLEW="$ROOT/gradlew"

if [ $THREADS -gt 0 ]; then
  FJ="-Djava.util.concurrent.ForkJoinPool.common.parallelism=$THREADS"
else
  FJ=""
fi

OUTDIR="$ROOT/build/reports"
mkdir -p "$OUTDIR"

for S in CAQR HOUSEHOLDER; do
  echo "Running microbench strategy=$S threads=$THREADS"
  "$GRADLEW" test --tests "net.faulj.benchmark.CAQRMicrobenchTest" -Dla.qr.strategy=$S $FJ
  SRC="$ROOT/build/reports/caqr_bench.csv"
  if [ -f "$SRC" ]; then
    STAMP=$(date +%Y%m%d%H%M%S)
    DEST="$ROOT/build/reports/caqr_bench_${S}_${STAMP}.csv"
    cp "$SRC" "$DEST"
    echo "Saved: $DEST"
  else
    echo "Microbench output not found for strategy $S"
  fi
done
