package net.faulj.compiler.matrix.cpu;

/**
 * Runtime choice for the CPU temporary-storage implementation.
 *
 * <p>The selector is deliberately an internal/developer switch. The public
 * matrix API does not expose it, and the default is the R1 reuse planner.</p>
 */
public enum MemoryPlannerStrategy {
    LEGACY("legacy"),
    REUSE("reuse");

    public static final String PROPERTY = "jlc.compiler.memoryPlanner";

    private final String propertyValue;

    MemoryPlannerStrategy(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }

    /**
     * Resolve the developer selector. An unset selector keeps the new planner
     * enabled; invalid values fail early instead of silently changing the A/B
     * experiment being run.
     */
    public static MemoryPlannerStrategy fromSystemProperty() {
        String configured = System.getProperty(PROPERTY);
        if (configured == null || configured.isBlank()) {
            return REUSE;
        }
        for (MemoryPlannerStrategy candidate : values()) {
            if (candidate.propertyValue.equalsIgnoreCase(configured.trim())) {
                return candidate;
            }
        }
        throw new IllegalArgumentException(
            "Unsupported " + PROPERTY + " value '" + configured
                + "'; expected legacy or reuse");
    }
}
