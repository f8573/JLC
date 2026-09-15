package net.faulj.nativeblas;

import java.util.Locale;

/**
 * Backend preference for dense compute dispatch.
 */
public enum BackendMode {
    JAVA("java"),
    NATIVE("native"),
    AUTO("auto");

    private final String id;

    BackendMode(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static BackendMode fromConfiguredValue(String value) {
        if (value == null || value.isBlank()) {
            return AUTO;
        }
        String normalized = value.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "java", "cpu" -> JAVA;
            case "native", "jni" -> NATIVE;
            case "auto" -> AUTO;
            default -> JAVA;
        };
    }
}
