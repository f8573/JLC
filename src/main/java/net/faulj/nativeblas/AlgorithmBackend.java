package net.faulj.nativeblas;

import java.util.Locale;

public enum AlgorithmBackend {
    AUTO,
    JAVA,
    CPP;

    static AlgorithmBackend fromConfiguredValue(String value) {
        if (value == null || value.isBlank()) {
            return AUTO;
        }
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "java", "disabled" -> JAVA;
            case "cpp", "c++" -> CPP;
            case "auto" -> AUTO;
            default -> AUTO;
        };
    }
}
