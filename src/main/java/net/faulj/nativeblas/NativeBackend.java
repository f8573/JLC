package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * JNI-backed backend facade with Java fallback for unsupported shapes/layouts.
 */
public final class NativeBackend implements ComputeBackend {
    private static final Logger LOGGER = Logger.getLogger(NativeBackend.class.getName());
    private static final String LIBRARY_NAME = "jlc_native";
    private static final int SVD_ALGORITHM_AUTO = 0;
    private static final int SVD_ALGORITHM_JACOBI = 1;
    private static final int SVD_ALGORITHM_BIDIAG_QR = 2;
    private static final AtomicReference<NativeContext> CONTEXT = new AtomicReference<>(NativeContext.notRequested());
    private static final Object LOAD_LOCK = new Object();

    private final ComputeBackend javaFallback;

    NativeBackend(ComputeBackend javaFallback) {
        this.javaFallback = Objects.requireNonNull(javaFallback, "javaFallback");
    }

    @Override
    public String backendId() {
        return "native";
    }

    @Override
    public boolean isAvailable() {
        return probe(true).isAvailable();
    }

    public NativeContext peekContext() {
        return CONTEXT.get();
    }

    public NativeContext probe(boolean shouldAttemptLoad) {
        NativeContext current = CONTEXT.get();
        if (!shouldAttemptLoad || current.getStatus() != NativeStatus.NOT_REQUESTED) {
            return current;
        }
        synchronized (LOAD_LOCK) {
            current = CONTEXT.get();
            if (current.getStatus() != NativeStatus.NOT_REQUESTED) {
                return current;
            }
            NativeContext next = attemptLoad();
            CONTEXT.set(next);
            return next;
        }
    }

    private NativeContext attemptLoad() {
        try {
            String libraryPath = findLibraryPath();
            if (libraryPath != null) {
                System.load(libraryPath);
            } else {
                System.loadLibrary(LIBRARY_NAME);
            }
            boolean available = NativeBindings.nativeIsAvailable();
            String runtimeDescription = NativeBindings.nativeRuntimeDescription();
            String providerDescription = NativeBindings.nativeProviderDescription();
            if (!available) {
                return new NativeContext(
                    false,
                    NativeStatus.LOAD_FAILED,
                    "Native library loaded but reported unavailable",
                    runtimeDescription,
                    providerDescription,
                    libraryPath,
                    NativeMatrixHandle.NULL
                );
            }
            long workspaceHandle = NativeBindings.nativeCreateContext(
                resolveDefaultThreadCount(),
                OffHeapMatrix.DEFAULT_ALIGNMENT,
                0
            );
            if (workspaceHandle == 0L) {
                return new NativeContext(
                    false,
                    NativeStatus.LOAD_FAILED,
                    "Native library loaded but failed to initialize workspace context",
                    runtimeDescription,
                    providerDescription,
                    libraryPath,
                    NativeMatrixHandle.NULL
                );
            }
            return new NativeContext(
                true,
                NativeStatus.READY,
                "Native JNI GEMM backend is available",
                runtimeDescription,
                providerDescription,
                libraryPath,
                new NativeMatrixHandle(workspaceHandle)
            );
        } catch (Throwable t) {
            String message = "Unable to load native library '" + LIBRARY_NAME + "': " + rootMessage(t);
            LOGGER.log(Level.WARNING, message);
            return new NativeContext(false, NativeStatus.LOAD_FAILED, message, null, null, null, NativeMatrixHandle.NULL);
        }
    }

    private static String findLibraryPath() {
        for (String candidate : candidateLibraryPaths()) {
            if (candidate == null || candidate.isBlank()) {
                continue;
            }
            Path path = Paths.get(candidate).toAbsolutePath().normalize();
            if (Files.exists(path)) {
                return path.toString();
            }
        }
        return null;
    }

    private static String[] candidateLibraryPaths() {
        String mappedName = System.mapLibraryName(LIBRARY_NAME);
        String userDir = System.getProperty("user.dir", ".");
        return new String[] {
            System.getProperty("jlc.native.lib.path"),
            System.getProperty("faulj.native.lib.path"),
            Paths.get(userDir, "build", "native-backend", "lib", mappedName).toString(),
            Paths.get(userDir, "native-backend", "build", "lib", mappedName).toString()
        };
    }

    private static String rootMessage(Throwable t) {
        Throwable current = t;
        while (current.getCause() != null && current.getCause() != current) {
            current = current.getCause();
        }
        String message = current.getMessage();
        if (message == null || message.isBlank()) {
            message = current.getClass().getSimpleName();
        }
        return message;
    }

    @Override
    public void gemm(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy) {
        if (isNativeDirectCompatible(a, b, c)) {
            gemmDirect((OffHeapMatrix) a, (OffHeapMatrix) b, (OffHeapMatrix) c, alpha, beta, policy);
            return;
        }
        if (usesOffHeapStorage(a, b, c) && isNativeHeapCompatible(a, b, c)) {
            gemmViaHeapMirror(a, b, c, alpha, beta, policy);
            return;
        }
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemm(a, b, c, alpha, beta, policy);
            return;
        }
        int threads = resolveThreadCount(policy == null ? DispatchPolicy.defaultPolicy() : policy);
        try {
            NativeBindings.nativeGemm(
                a.getRawData(), a.getRowCount(), a.getColumnCount(),
                b.getRawData(), b.getRowCount(), b.getColumnCount(),
                c.getRawData(), c.getRowCount(), c.getColumnCount(),
                alpha, beta,
                threads, nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemm(a, b, c, alpha, beta, policy);
        }
    }

    /**
     * Attempt the built-in native SVD. A null result asks the caller to use its
     * existing Java implementation, including after a native convergence or
     * linkage failure.
     */
    public SVDResult svd(Matrix a, boolean thin) {
        if (a == null || a.getRawImagData() != null) {
            return null;
        }
        NativeContext context = probe(true);
        if (!context.isAvailable()) {
            return null;
        }

        int m = a.getRowCount();
        int n = a.getColumnCount();
        int rank = Math.min(m, n);
        try {
            syncFromOffHeap(a);
            double[] uData = new double[m * m];
            double[] singularValues = new double[rank];
            double[] vData = new double[n * n];
            int nativeAlgorithm = resolveSvdAlgorithm(m, n);
            NativeBindings.nativeSvdDecomposeWithAlgorithm(
                a.getRawData(), m, n, uData, singularValues, vData, nativeAlgorithm);

            if (!isValidSvdOutput(uData, singularValues, vData, m, n)) {
                reportSvdFallback("native result failed structural validation", null);
                return null;
            }
            reportSvdAlgorithm(nativeAlgorithm);

            Matrix u = Matrix.wrap(uData, m, m);
            Matrix v = Matrix.wrap(vData, n, n);
            if (thin) {
                u = firstColumns(u, rank);
                v = firstColumns(v, rank);
            }
            return new SVDResult(a, u, singularValues, v);
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            reportSvdFallback("native invocation failed", ex);
            return null;
        }
    }
    @Override
    public void gemmStrided(double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
            return;
        }
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, 0,
                b, bOffset, ldb, k, n, 0,
                c, cOffset, ldc, m, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
        }
    }

    @Override
    public void gemmStrided(boolean transposeA,
                            double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStrided(transposeA, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
            return;
        }
        int aFlags = transposeA ? NativeFlags.A_TRANSPOSE : 0;
        int bRows = transposeA ? m : k;
        int cRows = transposeA ? k : m;
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, aFlags,
                b, bOffset, ldb, bRows, n, 0,
                c, cOffset, ldc, cRows, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStrided(transposeA, a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta);
        }
    }

    @Override
    public void gemmStrided(double[] a, int aOffset, int lda,
                            double[] b, int bOffset, int ldb,
                            double[] c, int cOffset, int ldc,
                            int m, int k, int n,
                            double alpha, double beta, int blockSize) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
            return;
        }
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, 0,
                b, bOffset, ldb, k, n, 0,
                c, cOffset, ldc, m, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStrided(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
        }
    }

    @Override
    public void gemmStridedTransA(double[] a, int aOffset, int lda,
                                  double[] b, int bOffset, int ldb,
                                  double[] c, int cOffset, int ldc,
                                  int m, int k, int n,
                                  double alpha, double beta, int blockSize) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStridedTransA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
            return;
        }
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, NativeFlags.A_TRANSPOSE,
                b, bOffset, ldb, m, n, 0,
                c, cOffset, ldc, k, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStridedTransA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
        }
    }

    @Override
    public void gemmStridedColMajorA(double[] a, int aOffset, int lda,
                                     double[] b, int bOffset, int ldb,
                                     double[] c, int cOffset, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStridedColMajorA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
            return;
        }
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, NativeFlags.A_COL_MAJOR,
                b, bOffset, ldb, k, n, 0,
                c, cOffset, ldc, m, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStridedColMajorA(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
        }
    }

    @Override
    public void gemmStridedColMajorB(double[] a, int aOffset, int lda,
                                     double[] b, int bOffset, int ldb,
                                     double[] c, int cOffset, int ldc,
                                     int m, int k, int n,
                                     double alpha, double beta, int blockSize) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStridedColMajorB(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
            return;
        }
        try {
            NativeBindings.nativeGemmStrided(
                a, aOffset, lda, m, k, 0,
                b, bOffset, ldb, k, n, NativeFlags.B_COL_MAJOR,
                c, cOffset, ldc, m, n, 0,
                alpha, beta,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStridedColMajorB(a, aOffset, lda, b, bOffset, ldb, c, cOffset, ldc, m, k, n, alpha, beta, blockSize);
        }
    }

    @Override
    public void gemmStridedBatched(double[] a, int aOffset, int lda, int aStride,
                                   double[] b, int bOffset, int ldb, int bStride,
                                   double[] c, int cOffset, int ldc, int cStride,
                                   int m, int k, int n,
                                   int batchCount,
                                   double alpha, double beta) {
        if (!isNativeCompatible(a, b, c)) {
            javaFallback.gemmStridedBatched(a, aOffset, lda, aStride,
                b, bOffset, ldb, bStride,
                c, cOffset, ldc, cStride,
                m, k, n, batchCount, alpha, beta);
            return;
        }
        try {
            NativeBindings.nativeGemmStridedBatched(
                a, aOffset, lda, m, k, 0, aStride,
                b, bOffset, ldb, k, n, 0, bStride,
                c, cOffset, ldc, m, n, 0, cStride,
                alpha, beta,
                batchCount,
                resolveDefaultThreadCount(), nativeExecutionFlags()
            );
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemmStridedBatched(a, aOffset, lda, aStride,
                b, bOffset, ldb, bStride,
                c, cOffset, ldc, cStride,
                m, k, n, batchCount, alpha, beta);
        }
    }

    static void resetForTests() {
        NativeContext context = CONTEXT.get();
        if (context != null && !context.getWorkspaceHandle().isNull()) {
            try {
                NativeBindings.nativeDestroyContext(context.getWorkspaceHandle().address());
            } catch (UnsatisfiedLinkError ignored) {
                // The native library was never loaded in this JVM, nothing to clean up.
            }
        }
        CONTEXT.set(NativeContext.notRequested());
    }

    private static boolean isNativeCompatible(Matrix a, Matrix b, Matrix c) {
        if (a == null || b == null || c == null) {
            return false;
        }
        return isNativeHeapCompatible(a, b, c) && !usesOffHeapStorage(a, b, c);
    }

    private static boolean isNativeHeapCompatible(Matrix a, Matrix b, Matrix c) {
        return a.getRawImagData() == null && b.getRawImagData() == null && c.getRawImagData() == null;
    }

    private void gemmDirect(OffHeapMatrix a, OffHeapMatrix b, OffHeapMatrix c,
                            double alpha, double beta, DispatchPolicy policy) {
        NativeContext context = probe(true);
        if (!context.isAvailable() || context.getWorkspaceHandle().isNull()) {
            javaFallback.gemm(a, b, c, alpha, beta, policy);
            return;
        }

        a.syncToOffHeap();
        b.syncToOffHeap();
        if (beta != 0.0) {
            c.syncToOffHeap();
        }

        int threads = resolveThreadCount(policy == null ? DispatchPolicy.defaultPolicy() : policy);
        try {
            NativeBindings.nativeGemmDirect(
                a.segment().asByteBuffer(), a.offsetBytes(), (int) a.ld(), a.rows(), a.cols(), nativeMatrixFlags(a, NativeMatrixRole.A),
                b.segment().asByteBuffer(), b.offsetBytes(), (int) b.ld(), b.rows(), b.cols(), nativeMatrixFlags(b, NativeMatrixRole.B),
                c.segment().asByteBuffer(), c.offsetBytes(), (int) c.ld(), c.rows(), c.cols(), nativeMatrixFlags(c, NativeMatrixRole.C),
                alpha, beta,
                threads, nativeExecutionFlags()
            );
            c.syncFromOffHeap();
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemm(a, b, c, alpha, beta, policy);
        }
    }

    private void gemmViaHeapMirror(Matrix a, Matrix b, Matrix c, double alpha, double beta, DispatchPolicy policy) {
        syncFromOffHeap(a);
        syncFromOffHeap(b);
        if (beta != 0.0) {
            syncFromOffHeap(c);
        }
        int threads = resolveThreadCount(policy == null ? DispatchPolicy.defaultPolicy() : policy);
        try {
            NativeBindings.nativeGemm(
                a.getRawData(), a.getRowCount(), a.getColumnCount(),
                b.getRawData(), b.getRowCount(), b.getColumnCount(),
                c.getRawData(), c.getRowCount(), c.getColumnCount(),
                alpha, beta,
                threads, nativeExecutionFlags()
            );
            syncToOffHeap(c);
        } catch (RuntimeException | UnsatisfiedLinkError ex) {
            javaFallback.gemm(a, b, c, alpha, beta, policy);
        }
    }

    private static void syncFromOffHeap(Matrix matrix) {
        if (matrix instanceof OffHeapMatrix offHeap) {
            offHeap.syncFromOffHeap();
        }
    }

    private static void syncToOffHeap(Matrix matrix) {
        if (matrix instanceof OffHeapMatrix offHeap) {
            offHeap.syncToOffHeap();
        }
    }

    private static boolean isNativeCompatible(double[] a, double[] b, double[] c) {
        return a != null && b != null && c != null;
    }

    private static Matrix firstColumns(Matrix matrix, int columnCount) {
        if (columnCount == matrix.getColumnCount()) {
            return matrix;
        }
        int rows = matrix.getRowCount();
        double[] data = new double[rows * columnCount];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(matrix.getRawData(), row * matrix.getColumnCount(),
                data, row * columnCount, columnCount);
        }
        return Matrix.wrap(data, rows, columnCount);
    }

    /**
     * Resolve the native SVD implementation in one place. The dimension cutoff
     * is intentionally an internal development policy, not a calibrated public
     * performance promise. It can be overridden by tests and benchmarks with
     * {@code jlc.algorithm.svd.nativeAlgorithm=jacobi|bidiag_qr}.
     */
    private static int resolveSvdAlgorithm(int rows, int cols) {
        String configured = System.getProperty("jlc.algorithm.svd.nativeAlgorithm");
        if (configured != null && !configured.isBlank()) {
            return switch (configured.trim().toLowerCase(java.util.Locale.ROOT)) {
                case "jacobi" -> SVD_ALGORITHM_JACOBI;
                case "bidiag_qr", "bidiagonal_qr", "bidiag-qr" -> SVD_ALGORITHM_BIDIAG_QR;
                case "auto" -> SVD_ALGORITHM_AUTO;
                default -> SVD_ALGORITHM_AUTO;
            };
        }
        int jacobiMaxDimension = nonNegativeIntProperty(
            "jlc.algorithm.svd.jacobiMaxDimension", 0);
        return Math.max(rows, cols) <= jacobiMaxDimension
            ? SVD_ALGORITHM_JACOBI : SVD_ALGORITHM_BIDIAG_QR;
    }

    private static int nonNegativeIntProperty(String name, int fallback) {
        String configured = System.getProperty(name);
        if (configured == null || configured.isBlank()) {
            return fallback;
        }
        try {
            int value = Integer.parseInt(configured.trim());
            return value >= 0 ? value : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static boolean isValidSvdOutput(double[] u, double[] singularValues, double[] v,
                                            int rows, int cols) {
        for (double value : u) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        for (double value : v) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        double previous = Double.POSITIVE_INFINITY;
        double orderTolerance = 128.0 * Math.ulp(1.0)
            * Math.max(1, Math.max(rows, cols));
        for (double value : singularValues) {
            if (!Double.isFinite(value) || value < 0.0) {
                return false;
            }
            if (value > previous && value - previous
                > orderTolerance * Math.max(1.0, Math.max(value, previous))) {
                return false;
            }
            previous = value;
        }
        return true;
    }

    private static void reportSvdFallback(String reason, Throwable cause) {
        if (!Boolean.getBoolean("jlc.algorithm.svd.debug")) {
            return;
        }
        if (cause == null) {
            LOGGER.log(Level.FINE, "Native SVD fallback: {0}", reason);
        } else {
            LOGGER.log(Level.FINE, "Native SVD fallback: " + reason, cause);
        }
    }

    private static void reportSvdAlgorithm(int algorithm) {
        if (!Boolean.getBoolean("jlc.algorithm.svd.debug")) {
            return;
        }
        String name = switch (algorithm) {
            case SVD_ALGORITHM_JACOBI -> "JACOBI";
            case SVD_ALGORITHM_BIDIAG_QR -> "BIDIAG_QR";
            default -> "AUTO (native default)";
        };
        LOGGER.log(Level.FINE, "Native SVD completed with algorithm: {0}", name);
    }
    private static boolean usesOffHeapStorage(Matrix a, Matrix b, Matrix c) {
        return a instanceof OffHeapMatrix || b instanceof OffHeapMatrix || c instanceof OffHeapMatrix;
    }

    private static boolean isNativeDirectCompatible(Matrix a, Matrix b, Matrix c) {
        if (!(a instanceof OffHeapMatrix offHeapA) || !(b instanceof OffHeapMatrix offHeapB) || !(c instanceof OffHeapMatrix offHeapC)) {
            return false;
        }
        if (offHeapA.getRawImagData() != null || offHeapB.getRawImagData() != null || offHeapC.getRawImagData() != null) {
            return false;
        }
        return true;
    }

    private static int nativeMatrixFlags(OffHeapMatrix matrix, NativeMatrixRole role) {
        int flags = 0;
        if (matrix.order() == OffHeapMatrix.Order.COL_MAJOR) {
            flags |= switch (role) {
                case A -> NativeFlags.A_COL_MAJOR;
                case B -> NativeFlags.B_COL_MAJOR;
                case C -> NativeFlags.C_COL_MAJOR;
            };
        }
        return flags;
    }

    private enum NativeMatrixRole {
        A,
        B,
        C
    }

    private static int resolveThreadCount(DispatchPolicy policy) {
        return policy.isParallelEnabled() ? policy.getParallelism() : 1;
    }

    private static int resolveDefaultThreadCount() {
        return resolveThreadCount(DispatchPolicy.defaultPolicy());
    }

    private static int nativeExecutionFlags() {
        return NativeFlags.FORCE_BUILTIN;
    }
}
