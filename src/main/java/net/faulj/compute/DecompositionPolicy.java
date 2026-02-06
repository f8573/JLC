package net.faulj.compute;

/**
 * Policy settings for block sizes and algorithm selection in decompositions.
 */
public final class DecompositionPolicy {
    private static final int DEFAULT_PANEL_SIZE = 64;
    private static final int DEFAULT_UPDATE_BLOCK_COLS = 128;
    private static final int DEFAULT_BLOCKED_THRESHOLD = 256;

    private static volatile DecompositionPolicy globalPolicy = builder().build();

    private final int panelSize;
    private final int updateBlockCols;
    private final int blockedThreshold;
    private final DispatchPolicy gemmPolicy;

    private DecompositionPolicy(Builder builder) {
        this.panelSize = builder.panelSize;
        this.updateBlockCols = builder.updateBlockCols;
        this.blockedThreshold = builder.blockedThreshold;
        this.gemmPolicy = builder.gemmPolicy == null ? DispatchPolicy.defaultPolicy() : builder.gemmPolicy;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DecompositionPolicy defaultPolicy() {
        return globalPolicy;
    }

    public static void setGlobalPolicy(DecompositionPolicy policy) {
        if (policy == null) {
            throw new IllegalArgumentException("Policy must not be null");
        }
        globalPolicy = policy;
    }

    public int panelSize(int rows, int cols) {
        int limit = Math.min(rows, cols);
        return Math.max(1, Math.min(panelSize, limit));
    }

    public int updateBlockCols(int rows, int cols) {
        int limit = Math.max(rows, cols);
        return Math.max(1, Math.min(updateBlockCols, limit));
    }

    public boolean useBlocked(int rows, int cols) {
        return Math.min(rows, cols) > blockedThreshold;
    }

    public int blockedThreshold() {
        return blockedThreshold;
    }

    public DispatchPolicy gemmPolicy() {
        return gemmPolicy;
    }

    public static final class Builder {
        private int panelSize = DEFAULT_PANEL_SIZE;
        private int updateBlockCols = DEFAULT_UPDATE_BLOCK_COLS;
        private int blockedThreshold = DEFAULT_BLOCKED_THRESHOLD;
        private DispatchPolicy gemmPolicy = DispatchPolicy.defaultPolicy();

        public Builder panelSize(int panelSize) {
            if (panelSize < 1) {
                throw new IllegalArgumentException("panelSize must be >= 1");
            }
            this.panelSize = panelSize;
            return this;
        }

        public Builder updateBlockCols(int updateBlockCols) {
            if (updateBlockCols < 1) {
                throw new IllegalArgumentException("updateBlockCols must be >= 1");
            }
            this.updateBlockCols = updateBlockCols;
            return this;
        }

        public Builder blockedThreshold(int blockedThreshold) {
            if (blockedThreshold < 1) {
                throw new IllegalArgumentException("blockedThreshold must be >= 1");
            }
            this.blockedThreshold = blockedThreshold;
            return this;
        }

        public Builder gemmPolicy(DispatchPolicy gemmPolicy) {
            this.gemmPolicy = gemmPolicy;
            return this;
        }

        public DecompositionPolicy build() {
            return new DecompositionPolicy(this);
        }
    }
}
