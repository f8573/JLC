package net.faulj.compiler.matrix;

/**
 * Floating-point reassociation permissions for matrix planning.
 *
 * <p>{@link #STRICT} preserves the association expressed by the user.
 * {@link #RELAXED} permits mathematically equivalent matrix-chain
 * reassociation, acknowledging that rounding can differ. {@link #FAST} is a
 * named future extension point and currently has the same M1 permissions as
 * {@code RELAXED}; it does not authorize additional unsafe rewrites yet.</p>
 */
public enum OptimizationSemantics {
    STRICT,
    RELAXED,
    FAST;

    /**
     * Whether M1 may choose another parenthesization for a pure MatMul chain.
     */
    public boolean allowsMatMulReassociation() {
        return this != STRICT;
    }
}
