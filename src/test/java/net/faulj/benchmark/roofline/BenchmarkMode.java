package net.faulj.benchmark.roofline;

import java.util.Locale;

final class BenchmarkMode {
    static final String MODE_PROPERTY = "jlc.benchmark.mode";
    static final String MODE_ENV = "JLC_BENCHMARK_MODE";

    private BenchmarkMode() {
    }

    static boolean isEnabled() {
        Boolean prop = parseFlag(System.getProperty(MODE_PROPERTY));
        if (prop != null) {
            return prop;
        }
        Boolean env = parseFlag(System.getenv(MODE_ENV));
        return env != null && env;
    }

    static Boolean parseFlag(String value) {
        if (value == null) {
            return null;
        }
        String v = value.trim().toLowerCase(Locale.ROOT);
        if (v.isEmpty()) {
            return null;
        }
        if ("1".equals(v) || "true".equals(v) || "yes".equals(v) || "on".equals(v)) {
            return Boolean.TRUE;
        }
        if ("0".equals(v) || "false".equals(v) || "no".equals(v) || "off".equals(v)) {
            return Boolean.FALSE;
        }
        return null;
    }
}
