package net.faulj.nativeblas;

/**
 * Shared bit flags for JNI GEMM layout and provider selection.
 */
final class NativeFlags {
    static final int A_TRANSPOSE = 1 << 0;
    static final int A_COL_MAJOR = 1 << 1;
    static final int B_TRANSPOSE = 1 << 2;
    static final int B_COL_MAJOR = 1 << 3;
    static final int C_COL_MAJOR = 1 << 4;
    static final int PREFER_VENDOR = 1 << 8;
    static final int FORCE_VENDOR = 1 << 9;
    static final int FORCE_BUILTIN = 1 << 10;

    private NativeFlags() {
    }
}
