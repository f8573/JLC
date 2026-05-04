package net.faulj.nativeblas;

public record NativeQrProfile(
    long calls,
    long wallNanos,
    long factorizeNanos,
    long inputTransposeNanos,
    long panelNanos,
    long reflectorPackNanos,
    long tBuildNanos,
    long trailingPackNanos,
    long trailingUnpackNanos,
    long trailingGemmNanos,
    long trailingTApplyNanos,
    long rExtractNanos,
    long qInitNanos,
    long qBuildNanos,
    long qGemmNanos,
    long qTApplyNanos
) {
    private static final int SNAPSHOT_FIELDS = 16;

    public static final NativeQrProfile EMPTY = new NativeQrProfile(
        0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L,
        0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L
    );

    static NativeQrProfile fromSnapshot(long[] snapshot) {
        if (snapshot == null || snapshot.length != SNAPSHOT_FIELDS) {
            return EMPTY;
        }
        return new NativeQrProfile(
            snapshot[0], snapshot[1], snapshot[2], snapshot[3],
            snapshot[4], snapshot[5], snapshot[6], snapshot[7],
            snapshot[8], snapshot[9], snapshot[10], snapshot[11],
            snapshot[12], snapshot[13], snapshot[14], snapshot[15]
        );
    }
}
