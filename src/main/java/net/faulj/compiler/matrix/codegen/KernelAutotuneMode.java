package net.faulj.compiler.matrix.codegen;

/** Explicit R5 operating mode. Tuning is never implicit in matrix execution. */
public enum KernelAutotuneMode {
    OFF("off"),
    PROFILE_ONLY("profile"),
    TUNE("tune"),
    USE_PROFILE("use");

    public static final String PROPERTY = "jlc.compiler.autotune";

    private final String propertyValue;

    KernelAutotuneMode(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }

    public boolean usesProfile() {
        return this == USE_PROFILE;
    }

    public boolean isExplicitTuningMode() {
        return this == TUNE || this == PROFILE_ONLY;
    }

    public static KernelAutotuneMode fromSystemProperty() {
        String configured = System.getProperty(PROPERTY);
        if (configured == null || configured.isBlank()) {
            return OFF;
        }
        for (KernelAutotuneMode mode : values()) {
            if (mode.propertyValue.equalsIgnoreCase(configured.trim())) {
                return mode;
            }
        }
        throw new IllegalArgumentException(
            "Unsupported " + PROPERTY + " value '" + configured
                + "'; expected off, profile, tune, or use");
    }
}
