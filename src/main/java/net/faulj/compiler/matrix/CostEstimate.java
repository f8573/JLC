package net.faulj.compiler.matrix;

/**
 * Saturation-safe cost summary for one operation or an entire plan.
 *
 * <p>The first field is deliberately named in terms of scalar
 * multiplications. M1 uses that quantity for matrix-chain selection; it is
 * not a claim about exact hardware FLOPs. The byte field is an estimate of
 * the operation's output materialization.</p>
 */
public record CostEstimate(long scalarMultiplications, long estimatedOutputBytes) {
    public static final CostEstimate ZERO = new CostEstimate(0L, 0L);

    public CostEstimate {
        if (scalarMultiplications < 0 || estimatedOutputBytes < 0) {
            throw new IllegalArgumentException("Cost values must be non-negative");
        }
    }

    public CostEstimate plus(CostEstimate other) {
        if (other == null) {
            throw new IllegalArgumentException("Cost estimate must not be null");
        }
        return new CostEstimate(
            SaturatingMath.add(scalarMultiplications, other.scalarMultiplications),
            SaturatingMath.add(estimatedOutputBytes, other.estimatedOutputBytes));
    }

    public long scalarMultiplicationCost() {
        return scalarMultiplications;
    }

    public long estimatedBytes() {
        return estimatedOutputBytes;
    }
}
