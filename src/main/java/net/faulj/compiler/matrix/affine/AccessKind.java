package net.faulj.compiler.matrix.affine;

/**
 * Memory effect of one affine access.
 */
public enum AccessKind {
    READ,
    WRITE,
    READ_WRITE,
    REDUCTION;

    public boolean reads() {
        return this == READ || this == READ_WRITE || this == REDUCTION;
    }

    public boolean writes() {
        return this == WRITE || this == READ_WRITE || this == REDUCTION;
    }

    public boolean isReduction() {
        return this == REDUCTION;
    }
}
