package net.faulj.compiler.matrix.cpu;

/**
 * Developer-selectable CPU fusion implementation.
 *
 * <p>The selector is intentionally separate from the historical M1-M4
 * semantics.  {@link #LEGACY} preserves the original M4 scale/add lowering,
 * {@link #GENERALIZED} enables the post-M4 fused-region planner, and
 * {@link #OFF} leaves every supported elementwise operation materialized.</p>
 */
public enum FusionStrategy {
    OFF("off"),
    LEGACY("legacy"),
    GENERALIZED("generalized");

    public static final String PROPERTY = "jlc.compiler.fusion";

    private final String propertyValue;

    FusionStrategy(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }

    /**
     * Resolve the developer switch.  The default remains the M4 path so that
     * existing benchmark and regression baselines remain reproducible until a
     * caller explicitly selects the R2 implementation.
     */
    public static FusionStrategy fromSystemProperty() {
        String configured = System.getProperty(PROPERTY);
        if (configured == null || configured.isBlank()) {
            return LEGACY;
        }
        String normalized = configured.trim();
        for (FusionStrategy candidate : values()) {
            if (candidate.propertyValue.equalsIgnoreCase(normalized)) {
                return candidate;
            }
        }
        throw new IllegalArgumentException(
            "Unsupported " + PROPERTY + " value '" + configured
                + "'; expected off, legacy, or generalized");
    }
}
