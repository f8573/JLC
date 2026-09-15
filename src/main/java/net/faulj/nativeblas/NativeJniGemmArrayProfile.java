package net.faulj.nativeblas;

/**
 * Immutable JNI-side profile for array-backed GEMM access timing.
 */
public record NativeJniGemmArrayProfile(
    long calls,
    long acquireNanos,
    long nativeNanos,
    long releaseNanos,
    long arrayAMod16,
    long arrayAMod32,
    long arrayAMod64,
    long arrayBMod16,
    long arrayBMod32,
    long arrayBMod64,
    long arrayCMod16,
    long arrayCMod32,
    long arrayCMod64,
    long directAMod16,
    long directAMod32,
    long directAMod64,
    long directBMod16,
    long directBMod32,
    long directBMod64,
    long directCMod16,
    long directCMod32,
    long directCMod64
) {
    private static final int SNAPSHOT_FIELDS = 22;
    public static final NativeJniGemmArrayProfile EMPTY = new NativeJniGemmArrayProfile(
        0L, 0L, 0L, 0L,
        0L, 0L, 0L,
        0L, 0L, 0L,
        0L, 0L, 0L,
        0L, 0L, 0L,
        0L, 0L, 0L,
        0L, 0L, 0L
    );

    static NativeJniGemmArrayProfile fromSnapshot(long[] snapshot) {
        if (snapshot == null || snapshot.length != SNAPSHOT_FIELDS) {
            return EMPTY;
        }
        return new NativeJniGemmArrayProfile(
            snapshot[0], snapshot[1], snapshot[2], snapshot[3],
            snapshot[4], snapshot[5], snapshot[6],
            snapshot[7], snapshot[8], snapshot[9],
            snapshot[10], snapshot[11], snapshot[12],
            snapshot[13], snapshot[14], snapshot[15],
            snapshot[16], snapshot[17], snapshot[18],
            snapshot[19], snapshot[20], snapshot[21]
        );
    }
}
