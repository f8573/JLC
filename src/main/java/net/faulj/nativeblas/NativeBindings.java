package net.faulj.nativeblas;

import java.nio.ByteBuffer;

/**
 * Raw JNI entry points implemented by the optional native backend library.
 */
final class NativeBindings {
    private NativeBindings() {
    }

    static native boolean nativeIsAvailable();

    static native boolean nativeVendorLapackAvailable();

    static native String nativeRuntimeDescription();

    static native String nativeProviderDescription();

    static native long nativeCreateContext(int preferredThreads, int alignmentBytes, int flags);

    static native void nativeDestroyContext(long contextHandle);

    static native void nativeProfileSetEnabled(boolean enabled);

    static native void nativeProfileReset();

    static native long[] nativeProfileSnapshot();

    static native void nativeJniGemmArrayProfileSetEnabled(boolean enabled);

    static native void nativeJniGemmArrayProfileReset();

    static native long[] nativeJniGemmArrayProfileSnapshot();

    static native void nativeGemmClearRuntimeOverrides();

    static native void nativeGemmSetRuntimeOverrides(int mc, int kc, int nc,
                                                     int mr, int nr,
                                                     boolean disableSquareTuning,
                                                     int loopOrder);

    static native void nativeQrProfileSetEnabled(boolean enabled);

    static native void nativeQrProfileReset();

    static native long[] nativeQrProfileSnapshot();

    static native void nativeLuProfileSetEnabled(boolean enabled);

    static native void nativeLuProfileReset();

    static native long[] nativeLuProfileSnapshot();

    static native void nativeQrSetBlockSizeOverride(int blockSize);

    static native void nativeQrSetGemmThreadsOverride(int threads);

    static native long nativeMatrixCreate(int rows, int cols, int order, int alignmentBytes);

    static native void nativeMatrixDestroy(long handle);

    static native ByteBuffer nativeMatrixBuffer(long handle);

    static native void nativeHessenbergReduceVendor(double[] h, int n);

    static native void nativeHessenbergDecomposeVendor(double[] h, int n, double[] q);

    static native void nativeHessenbergReduce(double[] h, int n);

    static native void nativeHessenbergDecompose(double[] h, int n, double[] q);

    static native void nativeBidiagonalDecompose(double[] a, int m, int n,
                                                 double[] u, double[] b, double[] v);

    static native void nativeLuFactor(double[] packedLu, int n, int[] pivots);

    static native void nativeLuFactorVendor(double[] packedLu, int n, int[] pivots);

    static native void nativeLuFactorDebug(double[] packedLu, int n, int[] pivots, boolean copyBack);

    static native void nativeQrFactorizeOnly(double[] a, int m, int n);

    static native void nativeQrFactorizeOnlyVendor(double[] a, int m, int n);

    static native void nativeQrDecompose(double[] a, int m, int n, int qCols,
                                         double[] q, double[] r);

    static native void nativeQrDecomposeVendor(double[] a, int m, int n, int qCols,
                                               double[] q, double[] r);

    static native int nativeCholeskyDecompose(double[] packedL, int n);

    static native int nativeCholeskyDecomposeVendor(double[] packedL, int n);

    static native void nativeGemm(double[] a, int aRows, int aCols,
                                  double[] b, int bRows, int bCols,
                                  double[] c, int cRows, int cCols,
                                  double alpha, double beta,
                                  int threads, int flags);

    static native void nativeGemmStrided(double[] a, int aOffset, int aLd, int aRows, int aCols, int aFlags,
                                         double[] b, int bOffset, int bLd, int bRows, int bCols, int bFlags,
                                         double[] c, int cOffset, int cLd, int cRows, int cCols, int cFlags,
                                         double alpha, double beta,
                                         int threads, int flags);

    static native void nativeGemmDirect(ByteBuffer aBuffer, long aByteOffset, int aLd, int aRows, int aCols, int aFlags,
                                        ByteBuffer bBuffer, long bByteOffset, int bLd, int bRows, int bCols, int bFlags,
                                        ByteBuffer cBuffer, long cByteOffset, int cLd, int cRows, int cCols, int cFlags,
                                        double alpha, double beta,
                                        int threads, int flags);

    static native void nativeGemmStridedBatched(double[] a, int aOffset, int aLd, int aRows, int aCols, int aFlags, int aStride,
                                                double[] b, int bOffset, int bLd, int bRows, int bCols, int bFlags, int bStride,
                                                double[] c, int cOffset, int cLd, int cRows, int cCols, int cFlags, int cStride,
                                                double alpha, double beta,
                                                int batchCount,
                                                int threads, int flags);
}
