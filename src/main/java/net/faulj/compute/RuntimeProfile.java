package net.faulj.compute;

/**
 * Applies runtime tuning profiles for constrained hardware.
 */
public final class RuntimeProfile {
    private static final String PROP_PROFILE = "faulj.runtime.profile";
    private static final String ENV_PROFILE = "FAULJ_RUNTIME_PROFILE";
    private static final String PROP_EXEC_POLICY = "faulj.exec.policy";
    private static final String ENV_EXEC_POLICY = "FAULJ_EXEC_POLICY";
    private static final String PROP_PARALLEL_ENABLED = "faulj.parallel.enabled";
    private static final String ENV_PARALLEL_ENABLED = "FAULJ_PARALLEL_ENABLED";
    private static final String PROP_PARALLELISM = "faulj.parallelism";
    private static final String ENV_PARALLELISM = "FAULJ_PARALLELISM";
    private static final String PROP_SIMD_ENABLED = "faulj.simd.enabled";
    private static final String ENV_SIMD_ENABLED = "FAULJ_SIMD_ENABLED";
    private static final String PROP_VECTORIZATION_ENABLED = "faulj.vectorization.enabled";
    private static final String ENV_VECTORIZATION_ENABLED = "FAULJ_VECTORIZATION_ENABLED";
    private static final String PROP_BLAS3_ENABLED = "faulj.blas3.enabled";
    private static final String ENV_BLAS3_ENABLED = "FAULJ_BLAS3_ENABLED";
    private static final String PROP_CUDA_ENABLED = "faulj.cuda.enabled";
    private static final String ENV_CUDA_ENABLED = "FAULJ_CUDA_ENABLED";

    private static volatile boolean applied;

    private RuntimeProfile() {
    }

    /**
     * Apply runtime profile once per JVM.
     * Supported profiles:
     * - default: no changes
     * - legacy: conservative settings for old/low-memory hardware
     */
    public static synchronized void applyConfiguredProfile() {
        if (applied) {
            return;
        }
        String raw = System.getProperty(PROP_PROFILE);
        if (raw == null || raw.isBlank()) {
            raw = System.getenv(ENV_PROFILE);
        }
        String profile = (raw == null ? "default" : raw.trim().toLowerCase());
        if ("legacy".equals(profile)) {
            applyLegacyProfile();
        }
        applyExecutionPolicyFlags();
        applied = true;
    }

    private static void applyLegacyProfile() {
        // Conservative defaults for old/ARM/no-AVX systems:
        // single thread, scalar kernels, no CUDA, small blocks.
        DispatchPolicy safePolicy = DispatchPolicy.builder()
                .enableParallel(false)
                .parallelism(1)
                .parallelThreshold(Integer.MAX_VALUE)
                .enableCuda(false)
                .enableSimd(false)
                .simdThreshold(Integer.MAX_VALUE)
                .enableBlas3(false)
                .blas3Threshold(Integer.MAX_VALUE)
                .enableStrassen(false)
                .naiveThreshold(1024)
                .blockedThreshold(Integer.MAX_VALUE)
                .blockSize(16)
                .minBlockSize(8)
                .maxBlockSize(32)
                .build();
        DispatchPolicy.setGlobalPolicy(safePolicy);

        DecompositionPolicy.setGlobalPolicy(
                DecompositionPolicy.builder()
                        .panelSize(16)
                        .updateBlockCols(32)
                        .blockedThreshold(1024)
                        .gemmPolicy(safePolicy)
                        .build()
        );

        // Keep lower layers consistent with the profile.
        System.setProperty("faulj.cuda.enabled", "false");
        System.setProperty("faulj.simd.enabled", "false");
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "1");
        if (System.getProperty("faulj.benchmark.isolated.maxHeapMb") == null) {
            System.setProperty("faulj.benchmark.isolated.maxHeapMb", "192");
        }
    }

    private enum ExecPolicy {
        AUTO,
        SCALAR_SAFE,
        SCALAR_PARALLEL,
        SIMD,
        ACCEL
    }

    private static void applyExecutionPolicyFlags() {
        ExecPolicy execPolicy = parseExecPolicy(readText(PROP_EXEC_POLICY, ENV_EXEC_POLICY));
        int available = Math.max(1, Runtime.getRuntime().availableProcessors());

        Boolean simdFlag = readBoolean(PROP_SIMD_ENABLED, ENV_SIMD_ENABLED);
        Boolean vecFlag = readBoolean(PROP_VECTORIZATION_ENABLED, ENV_VECTORIZATION_ENABLED);
        boolean simdEnabled = simdFlag != null ? simdFlag : (vecFlag != null ? vecFlag : true);

        Boolean parallelFlag = readBoolean(PROP_PARALLEL_ENABLED, ENV_PARALLEL_ENABLED);
        boolean parallelEnabled = parallelFlag == null || parallelFlag;
        int parallelism = clampParallelism(readInt(PROP_PARALLELISM, ENV_PARALLELISM, available), available);

        Boolean blas3Flag = readBoolean(PROP_BLAS3_ENABLED, ENV_BLAS3_ENABLED);
        boolean blas3Enabled = blas3Flag == null ? simdEnabled : blas3Flag;

        Boolean cudaFlag = readBoolean(PROP_CUDA_ENABLED, ENV_CUDA_ENABLED);
        boolean cudaEnabled = cudaFlag != null && cudaFlag;

        int naiveThreshold = 64;
        int blockedThreshold = 256;
        int parallelThreshold = 2048;
        int blas3Threshold = 256;
        int blockSize = 128;
        int minBlockSize = 128;
        int maxBlockSize = 256;

        switch (execPolicy) {
            case SCALAR_SAFE -> {
                parallelEnabled = false;
                parallelism = 1;
                simdEnabled = false;
                blas3Enabled = false;
                cudaEnabled = false;
                naiveThreshold = Integer.MAX_VALUE;
                blockedThreshold = Integer.MAX_VALUE;
                parallelThreshold = Integer.MAX_VALUE;
                blas3Threshold = Integer.MAX_VALUE;
                blockSize = 16;
                minBlockSize = 8;
                maxBlockSize = 32;
            }
            case SCALAR_PARALLEL -> {
                parallelEnabled = true;
                if (parallelism < 1) parallelism = available;
                simdEnabled = false;
                blas3Enabled = false;
                cudaEnabled = false;
                naiveThreshold = Integer.MAX_VALUE;
                blockedThreshold = Integer.MAX_VALUE;
                parallelThreshold = 1;
                blas3Threshold = Integer.MAX_VALUE;
                blockSize = 16;
                minBlockSize = 8;
                maxBlockSize = 32;
            }
            case SIMD -> {
                simdEnabled = true;
                blas3Enabled = true;
                cudaEnabled = false;
            }
            case ACCEL -> {
                simdEnabled = true;
                blas3Enabled = true;
                cudaEnabled = true;
            }
            case AUTO -> {
                // Respect explicit flags and use adaptive defaults.
            }
        }

        DispatchPolicy policy = DispatchPolicy.builder()
                .enableParallel(parallelEnabled)
                .parallelism(Math.max(1, parallelism))
                .parallelThreshold(parallelThreshold)
                .enableCuda(cudaEnabled)
                .enableSimd(simdEnabled)
                .simdThreshold(256)
                .enableBlas3(blas3Enabled)
                .blas3Threshold(blas3Threshold)
                .enableStrassen(false)
                .naiveThreshold(naiveThreshold)
                .blockedThreshold(blockedThreshold)
                .blockSize(blockSize)
                .minBlockSize(minBlockSize)
                .maxBlockSize(maxBlockSize)
                .build();
        DispatchPolicy.setGlobalPolicy(policy);

        DecompositionPolicy.setGlobalPolicy(
                DecompositionPolicy.builder()
                        .panelSize(Math.min(64, Math.max(8, blockSize)))
                        .updateBlockCols(Math.min(128, Math.max(16, blockSize * 2)))
                        .blockedThreshold(blockedThreshold == Integer.MAX_VALUE ? 1024 : 256)
                        .gemmPolicy(policy)
                        .build()
        );

        // Propagate effective flags to lower-level detectors.
        System.setProperty(PROP_SIMD_ENABLED, Boolean.toString(simdEnabled));
        System.setProperty(PROP_CUDA_ENABLED, Boolean.toString(cudaEnabled));
        System.setProperty(PROP_PARALLEL_ENABLED, Boolean.toString(parallelEnabled));
        System.setProperty(PROP_PARALLELISM, Integer.toString(Math.max(1, parallelism)));
    }

    private static ExecPolicy parseExecPolicy(String raw) {
        if (raw == null || raw.isBlank()) {
            return ExecPolicy.AUTO;
        }
        try {
            return ExecPolicy.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            return ExecPolicy.AUTO;
        }
    }

    private static int clampParallelism(int p, int available) {
        if (p <= 0) return available;
        return Math.max(1, Math.min(p, available));
    }

    private static String readText(String prop, String env) {
        String v = System.getProperty(prop);
        if (v == null || v.isBlank()) {
            v = System.getenv(env);
        }
        return v;
    }

    private static int readInt(String prop, String env, int fallback) {
        String raw = readText(prop, env);
        if (raw == null || raw.isBlank()) return fallback;
        try {
            return Integer.parseInt(raw.trim());
        } catch (Exception ex) {
            return fallback;
        }
    }

    private static Boolean readBoolean(String prop, String env) {
        String raw = readText(prop, env);
        if (raw == null || raw.isBlank()) return null;
        String v = raw.trim().toLowerCase();
        if ("1".equals(v) || "true".equals(v) || "yes".equals(v) || "on".equals(v) || "y".equals(v)) {
            return Boolean.TRUE;
        }
        if ("0".equals(v) || "false".equals(v) || "no".equals(v) || "off".equals(v) || "n".equals(v)) {
            return Boolean.FALSE;
        }
        return null;
    }
}
