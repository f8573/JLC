package net.faulj.compute;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Locale;

/**
 * Utility for detecting Java Vector API availability.
 */
final class VectorSupport {
    private static final String PROP_FORCE = "faulj.simd.force";
    private static final String PROP_ENABLED = "faulj.simd.enabled";
    private static final String PROP_AVAILABLE = "faulj.simd.available";
    private static final String PROP_ALLOW_NON_X86 = "faulj.simd.allow_non_x86";
    private static final String ENV_ENABLED = "FAULJ_SIMD_ENABLED";
    private static final String ENV_AVAILABLE = "FAULJ_SIMD_AVAILABLE";
    private static final String ENV_ALLOW_NON_X86 = "FAULJ_SIMD_ALLOW_NON_X86";
    private static final String VECTOR_CLASS = "jdk.incubator.vector.DoubleVector";

    private static volatile Boolean cached;

    private VectorSupport() {
    }

    static boolean isVectorApiAvailable() {
        Boolean local = cached;
        if (local != null) {
            return local;
        }
        boolean available = detect();
        cached = available;
        return available;
    }

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

        // Conservative default: disable SIMD on non-x86 unless explicitly allowed.
        // This avoids AVX-assumptive paths on older/ARM hardware.
        if (!isLikelyX86() && !isExplicitlyAllowNonX86()) {
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

        return hasVectorApi();
    }

    private static boolean isLikelyX86() {
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("x86") || arch.contains("amd64");
    }

    private static boolean isExplicitlyAllowNonX86() {
        Boolean prop = readFlag(System.getProperty(PROP_ALLOW_NON_X86));
        if (prop != null) {
            return prop;
        }
        Boolean env = readFlag(System.getenv(ENV_ALLOW_NON_X86));
        return env != null && env;
    }

    private static boolean isDisabled(String propValue, String envValue) {
        Boolean prop = readFlag(propValue);
        if (prop != null && !prop) {
            return true;
        }
        Boolean env = readFlag(envValue);
        return env != null && !env;
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

    private static boolean hasVectorApi() {
        try {
            Class<?> vectorClass = Class.forName(VECTOR_CLASS, false, VectorSupport.class.getClassLoader());
            Field speciesField = vectorClass.getField("SPECIES_PREFERRED");
            Object species = speciesField.get(null);
            Method lengthMethod = species.getClass().getMethod("length");
            int length = (int) lengthMethod.invoke(species);
            return length > 1;
        } catch (Throwable ex) {
            return false;
        }
    }
}
