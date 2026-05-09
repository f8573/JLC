package net.faulj.nativeblas;

/**
 * Immutable snapshot of the native GEMM profiler counters.
 */
public record NativeGemmProfile(
    long calls,
    long wallNanos,
    long vendorCalls,
    long vendorNanos,
    long scaleCNanos,
    long packANanos,
    long packBNanos,
    long kernelNanos,
    long threadLaunchNanos,
    long threadJoinNanos,
    long packACalls,
    long packBCalls,
    long microtileCalls,
    long packABytes,
    long packBBytes,
    long lastRequestedThreads,
    long lastActualThreads,
    long lastPanelCount,
    long lastMc,
    long lastKc,
    long lastNc,
    long lastMr,
    long lastNr,
    long allocNanos,
    long jcLoopNanos,
    long pcLoopNanos,
    long icLoopNanos,
    long edgeNanos,
    long jcPanels,
    long pcPanels,
    long icBlocks
) {
    private static final int SNAPSHOT_FIELDS = 31;
    public static final NativeGemmProfile EMPTY = new NativeGemmProfile(
        0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
        0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L
    );

    static NativeGemmProfile fromSnapshot(long[] snapshot) {
        if (snapshot == null || snapshot.length != SNAPSHOT_FIELDS) {
            return EMPTY;
        }
        return new NativeGemmProfile(
            snapshot[0],
            snapshot[1],
            snapshot[2],
            snapshot[3],
            snapshot[4],
            snapshot[5],
            snapshot[6],
            snapshot[7],
            snapshot[8],
            snapshot[9],
            snapshot[10],
            snapshot[11],
            snapshot[12],
            snapshot[13],
            snapshot[14],
            snapshot[15],
            snapshot[16],
            snapshot[17],
            snapshot[18],
            snapshot[19],
            snapshot[20],
            snapshot[21],
            snapshot[22],
            snapshot[23],
            snapshot[24],
            snapshot[25],
            snapshot[26],
            snapshot[27],
            snapshot[28],
            snapshot[29],
            snapshot[30]
        );
    }

    public long totalProfileNanos() {
        return vendorNanos + scaleCNanos + packANanos + packBNanos + kernelNanos + threadLaunchNanos + threadJoinNanos;
    }

    public double wallSeconds() {
        return wallNanos / 1e9;
    }

    public long nativeKernelNanos() {
        return vendorCalls > 0 ? vendorNanos : kernelNanos;
    }

    public long nativeThreadingNanos() {
        return threadLaunchNanos + threadJoinNanos;
    }

    public boolean hasTimingData() {
        return calls > 0 && wallNanos > 0;
    }
}
