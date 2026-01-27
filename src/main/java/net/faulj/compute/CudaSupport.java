package net.faulj.compute;

import java.util.Locale;

/**
 * Utility for detecting CUDA runtime availability.
 */
final class CudaSupport {
    private static final String PROP_FORCE = "faulj.cuda.force";
    private static final String PROP_ENABLED = "faulj.cuda.enabled";
    private static final String PROP_AVAILABLE = "faulj.cuda.available";
    private static final String ENV_ENABLED = "FAULJ_CUDA_ENABLED";
    private static final String ENV_AVAILABLE = "FAULJ_CUDA_AVAILABLE";
    private static final String ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES";
    private static final String ENV_NVIDIA_VISIBLE_DEVICES = "NVIDIA_VISIBLE_DEVICES";
    private static final String[] CUDA_CLASSES = {
            "jcuda.runtime.JCuda",
            "jcuda.driver.JCudaDriver",
            "jcuda.jcublas.JCublas2",
            "org.bytedeco.cuda.global.cudart",
            "ai.djl.cuda.CudaUtils"
    };

    private static volatile Boolean cached;

    private CudaSupport() {
    }

    /**
     * Check whether CUDA is available in the current runtime.
     *
     * @return true if CUDA is available
     */
    static boolean isCudaAvailable() {
        Boolean local = cached;
        if (local != null) {
            return local;
        }
        boolean available = detect();
        cached = available;
        return available;
    }

    /**
     * Reset cached CUDA detection state.
     */
    static void refresh() {
        cached = null;
    }

    private static boolean detect() {
        Boolean forced = readFlag(System.getProperty(PROP_FORCE));
        if (forced != null) {
            return forced;
        }

        if (isDisabled(System.getProperty(PROP_ENABLED), System.getenv(ENV_ENABLED))) {
            return false;
        }

        Boolean explicitAvailable = readFlag(System.getProperty(PROP_AVAILABLE));
        if (explicitAvailable != null) {
            return explicitAvailable;
        }
        explicitAvailable = readFlag(System.getenv(ENV_AVAILABLE));
        if (explicitAvailable != null) {
            return explicitAvailable;
        }

        if (isHiddenByEnv(System.getenv(ENV_CUDA_VISIBLE_DEVICES))
                || isHiddenByEnv(System.getenv(ENV_NVIDIA_VISIBLE_DEVICES))) {
            return false;
        }

        return hasCudaRuntime();
    }

    private static boolean isDisabled(String propValue, String envValue) {
        Boolean prop = readFlag(propValue);
        if (prop != null && !prop) {
            return true;
        }
        Boolean env = readFlag(envValue);
        return env != null && !env;
    }

    private static boolean isHiddenByEnv(String value) {
        if (value == null) {
            return false;
        }
        String v = value.trim();
        if (v.isEmpty()) {
            return true;
        }
        String lower = v.toLowerCase(Locale.ROOT);
        return "none".equals(lower) || "void".equals(lower) || "-1".equals(lower);
    }

    private static Boolean readFlag(String value) {
        if (value == null) {
            return null;
        }
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.isEmpty()) {
            return null;
        }
        if ("1".equals(v) || "true".equals(v) || "yes".equals(v) || "y".equals(v) || "on".equals(v)) {
            return Boolean.TRUE;
        }
        if ("0".equals(v) || "false".equals(v) || "no".equals(v) || "n".equals(v) || "off".equals(v)) {
            return Boolean.FALSE;
        }
        return null;
    }

    private static boolean hasCudaRuntime() {
        for (String className : CUDA_CLASSES) {
            if (classPresent(className)) {
                return true;
            }
        }
        return false;
    }

    private static boolean classPresent(String className) {
        try {
            Class.forName(className, false, CudaSupport.class.getClassLoader());
            return true;
        } catch (ClassNotFoundException ex) {
            return false;
        }
    }
}
