package net.faulj.nativeblas;

/**
 * Lightweight native resource handle wrapper used by backend diagnostics.
 */
public record NativeMatrixHandle(long address) {
    public static final NativeMatrixHandle NULL = new NativeMatrixHandle(0L);

    public boolean isNull() {
        return address == 0L;
    }
}
