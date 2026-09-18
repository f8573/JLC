package net.faulj.autotune.probe.memory;

import net.faulj.autotune.probe.ProbeConfidence;

import java.util.ArrayList;
import java.util.List;

public final class CacheSizeProbe {

    public static class CacheSweepPoint {
        public final int bytes;
        public final double bandwidthBytesPerSec;

        CacheSweepPoint(int bytes, double bandwidthBytesPerSec) {
            this.bytes = bytes;
            this.bandwidthBytesPerSec = bandwidthBytesPerSec;
        }
    }

    // Sweep sizes doubling from 1KB to 64MB
    public static List<CacheSweepPoint> sweep() {
        List<CacheSweepPoint> out = new ArrayList<>();
        int start = 1024; // 1KB
        int max = 64 * 1024 * 1024; // 64MB
        for (int sz = start; sz <= max; sz *= 2) {
            int iters = selectIters(sz);
            double bw = BandwidthProbe.measureTriadBytesPerSec(sz, 3, 2);
            out.add(new CacheSweepPoint(sz, bw));
        }
        return out;
    }

    private static int selectIters(int size) {
        if (size <= 8 * 1024) return 10;
        if (size <= 256 * 1024) return 6;
        return 3;
    }

    // Heuristic detection: find sizes where bandwidth drops below 70% of peak seen so far
    public static long detectEffectiveL1Bytes(List<CacheSweepPoint> sweep) {
        double peak = 0.0;
        for (CacheSweepPoint p : sweep) {
            if (p.bandwidthBytesPerSec > peak) peak = p.bandwidthBytesPerSec;
            if (p.bandwidthBytesPerSec < peak * 0.70) {
                return p.bytes;
            }
        }
        return 32768L; // default 32KB
    }

    public static long detectEffectiveL2Bytes(List<CacheSweepPoint> sweep) {
        double peak = 0.0;
        boolean passedL1 = false;
        for (CacheSweepPoint p : sweep) {
            if (!passedL1 && p.bytes >= detectEffectiveL1Bytes(sweep)) passedL1 = true;
            if (p.bandwidthBytesPerSec > peak) peak = p.bandwidthBytesPerSec;
            if (passedL1 && p.bandwidthBytesPerSec < peak * 0.70) {
                return p.bytes;
            }
        }
        return 256 * 1024L; // default 256KB
    }

    public static long detectEffectiveL3Bytes(List<CacheSweepPoint> sweep) {
        double peak = 0.0;
        boolean passedL2 = false;
        long l2 = detectEffectiveL2Bytes(sweep);
        for (CacheSweepPoint p : sweep) {
            if (!passedL2 && p.bytes >= l2) passedL2 = true;
            if (p.bandwidthBytesPerSec > peak) peak = p.bandwidthBytesPerSec;
            if (passedL2 && p.bandwidthBytesPerSec < peak * 0.70) {
                return p.bytes;
            }
        }
        return 8L * 1024 * 1024; // default 8MB
    }

    public static ProbeConfidence defaultConfidence() {
        return ProbeConfidence.MEASURED;
    }

    /**
     * Run the cache size probe: sweep bandwidth across sizes and detect cache boundaries.
     */
    public static CacheSizeResult run() {
        try {
            List<CacheSweepPoint> sweepData = sweep();
            long l1 = detectEffectiveL1Bytes(sweepData);
            long l2 = detectEffectiveL2Bytes(sweepData);
            long l3 = detectEffectiveL3Bytes(sweepData);
            return new CacheSizeResult(l1, l2, l3, ProbeConfidence.MEASURED);
        } catch (Exception e) {
            return new CacheSizeResult(32768L, 256 * 1024L, 8L * 1024 * 1024,
                    ProbeConfidence.FAILED);
        }
    }

    public static final class CacheSizeResult {
        public final long l1Bytes;
        public final long l2Bytes;
        public final long l3Bytes;
        public final ProbeConfidence confidence;

        public CacheSizeResult(long l1Bytes, long l2Bytes, long l3Bytes,
                               ProbeConfidence confidence) {
            this.l1Bytes = l1Bytes;
            this.l2Bytes = l2Bytes;
            this.l3Bytes = l3Bytes;
            this.confidence = confidence;
        }
    }
}
