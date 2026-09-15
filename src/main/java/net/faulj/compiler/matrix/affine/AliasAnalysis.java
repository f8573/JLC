package net.faulj.compiler.matrix.affine;

/**
 * Conservative alias relation for logical buffers.
 */
public final class AliasAnalysis {
    private AliasAnalysis() {
    }

    /**
     * Return the strongest relation justified by logical buffer provenance.
     * Memory-space differences do not prove that two external buffers are
     * disjoint because wrappers and shared backing storage are possible.
     */
    public static AliasRelation between(LogicalBuffer left, LogicalBuffer right) {
        if (left == null || right == null) {
            throw new IllegalArgumentException("Buffers must not be null");
        }
        if (left == right) {
            return AliasRelation.MUST_ALIAS;
        }
        if (left.isTemporary() || right.isTemporary()) {
            return AliasRelation.NO_ALIAS;
        }
        if (left.isSymbolic() || right.isSymbolic()) {
            return AliasRelation.NO_ALIAS;
        }
        // Distinct external Matrix objects may still share backing storage.
        return AliasRelation.MAY_ALIAS;
    }

    public static AliasRelation relation(LogicalBuffer left, LogicalBuffer right) {
        return between(left, right);
    }

    public static AliasRelation analyze(LogicalBuffer left, LogicalBuffer right) {
        return between(left, right);
    }
}
