package net.faulj.compiler.matrix.affine;

import java.util.Objects;

/**
 * Explicit metadata for a reduction statement.
 */
public record ReductionMetadata(AffineVariable variable, ReductionSemantics semantics) {
    public ReductionMetadata {
        Objects.requireNonNull(variable, "Reduction variable must not be null");
        Objects.requireNonNull(semantics, "Reduction semantics must not be null");
    }

    public boolean allowsReassociation() {
        return semantics.allowsReassociation();
    }

    @Override
    public String toString() {
        return variable + " (" + semantics + ")";
    }
}
