package net.faulj.nativeblas;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Central backend selection and diagnostics registry.
 */
public final class BackendRegistry {
    private static final Logger LOGGER = Logger.getLogger(BackendRegistry.class.getName());
    private static final ComputeBackend JAVA_BACKEND = new JavaBackend();
    private static final NativeBackend NATIVE_BACKEND = new NativeBackend(JAVA_BACKEND);
    private static final AtomicBoolean FALLBACK_LOGGED = new AtomicBoolean(false);

    private BackendRegistry() {
    }

    public static ComputeBackend gemmBackend() {
        return selectBackend(true);
    }

    public static ComputeBackend gemmBackend(int rows, int cols, int threadCount) {
        return algorithmBackend(new AlgorithmDispatchRequest("gemm", "multiply", rows, cols, threadCount), true);
    }

    public static ComputeBackend gemmBackend(String mode, int rows, int cols, int threadCount) {
        return algorithmBackend(new AlgorithmDispatchRequest("gemm", mode, rows, cols, threadCount), true);
    }

    static boolean shouldUseCppForAlgorithm(String algorithm, String mode, int rows, int cols, int threadCount) {
        BackendMode requested = requestedBackend();
        NativeContext nativeContext = nativeContextFor(requested);
        return requested != BackendMode.JAVA
            && AlgorithmDispatch.shouldUseCpp(new AlgorithmDispatchRequest(algorithm, mode, rows, cols, threadCount), nativeContext);
    }

    public static BackendSnapshot snapshot() {
        BackendMode requested = requestedBackend();
        NativeContext nativeContext = nativeContextFor(requested);
        ComputeBackend active = selectBackend(false, requested, nativeContext);
        boolean fallbackToJava = requested != BackendMode.JAVA && "java".equals(active.backendId());
        return new BackendSnapshot(requested, active.backendId(), fallbackToJava, nativeContext);
    }

    private static ComputeBackend selectBackend(boolean logFallback) {
        BackendMode requested = requestedBackend();
        NativeContext nativeContext = nativeContextFor(requested);
        return selectBackend(logFallback, requested, nativeContext);
    }

    private static ComputeBackend selectBackend(boolean logFallback, BackendMode requested, NativeContext nativeContext) {
        if ((requested == BackendMode.NATIVE || requested == BackendMode.AUTO) && nativeContext.isAvailable()) {
            return NATIVE_BACKEND;
        }
        if (logFallback && requested == BackendMode.NATIVE && FALLBACK_LOGGED.compareAndSet(false, true)) {
            LOGGER.log(Level.INFO, "jlc.backend=native requested, falling back to java backend: {0}", nativeContext.getMessage());
        }
        return JAVA_BACKEND;
    }

    private static ComputeBackend algorithmBackend(AlgorithmDispatchRequest request, boolean logFallback) {
        BackendMode requested = requestedBackend();
        NativeContext nativeContext = nativeContextFor(requested);
        if (requested != BackendMode.JAVA && AlgorithmDispatch.shouldUseCpp(request, nativeContext)) {
            return NATIVE_BACKEND;
        }
        if (logFallback && requested == BackendMode.NATIVE && !nativeContext.isAvailable()
            && FALLBACK_LOGGED.compareAndSet(false, true)) {
            LOGGER.log(Level.INFO, "jlc.backend=native requested, falling back to java backend: {0}", nativeContext.getMessage());
        }
        return JAVA_BACKEND;
    }

    private static NativeContext nativeContextFor(BackendMode requested) {
        boolean shouldProbe = requested == BackendMode.NATIVE || requested == BackendMode.AUTO;
        return NATIVE_BACKEND.probe(shouldProbe);
    }

    static BackendMode requestedBackend() {
        String configured = firstNonBlank(
            System.getProperty("jlc.backend"),
            System.getProperty("faulj.backend"),
            System.getenv("JLC_BACKEND"),
            System.getenv("FAULJ_BACKEND")
        );
        return BackendMode.fromConfiguredValue(configured);
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }

    static void resetForTests() {
        FALLBACK_LOGGED.set(false);
        AlgorithmDispatch.resetForTests();
        NativeBackend.resetForTests();
    }
}
