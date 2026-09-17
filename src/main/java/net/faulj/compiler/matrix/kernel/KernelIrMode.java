package net.faulj.compiler.matrix.kernel;

/** Developer selector for the optional R3 CPU integration. */
public enum KernelIrMode {
    OFF("off"),
    VERIFY("verify"),
    EXECUTE("execute");

    public static final String PROPERTY = "jlc.compiler.kernelIr";

    private final String propertyValue;

    KernelIrMode(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }

    public boolean isEnabled() {
        return this != OFF;
    }

    public static KernelIrMode fromSystemProperty() {
        String configured = System.getProperty(PROPERTY);
        if (configured == null || configured.isBlank()) {
            return OFF;
        }
        for (KernelIrMode candidate : values()) {
            if (candidate.propertyValue.equalsIgnoreCase(configured.trim())) {
                return candidate;
            }
        }
        throw new IllegalArgumentException(
            "Unsupported " + PROPERTY + " value '" + configured
                + "'; expected off, verify, or execute");
    }
}
