package net.faulj.nativeblas;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Thread-local algorithm backend overrides used by higher-level hybrid algorithms.
 */
public final class NativeAlgorithmScope {
    private static final ThreadLocal<Map<String, AlgorithmBackend>> OVERRIDES =
        ThreadLocal.withInitial(Collections::emptyMap);

    private NativeAlgorithmScope() {
    }

    static AlgorithmBackend overrideFor(String algorithm) {
        if (algorithm == null || algorithm.isBlank()) {
            return null;
        }
        return OVERRIDES.get().get(normalize(algorithm));
    }

    public static <T> T withOverride(String algorithm, AlgorithmBackend backend, Supplier<T> supplier) {
        Map<String, AlgorithmBackend> overrides = new HashMap<>();
        overrides.put(normalize(algorithm), backend);
        return withOverrides(overrides, supplier);
    }

    public static <T> T withOverrides(Map<String, AlgorithmBackend> overrides, Supplier<T> supplier) {
        Map<String, AlgorithmBackend> previous = OVERRIDES.get();
        Map<String, AlgorithmBackend> next = new HashMap<>(previous);
        overrides.forEach((algorithm, backend) -> next.put(normalize(algorithm), backend));
        OVERRIDES.set(Collections.unmodifiableMap(next));
        try {
            return supplier.get();
        } finally {
            OVERRIDES.set(previous);
        }
    }

    private static String normalize(String algorithm) {
        return algorithm.trim().toLowerCase(java.util.Locale.ROOT).replace('-', '_');
    }
}
