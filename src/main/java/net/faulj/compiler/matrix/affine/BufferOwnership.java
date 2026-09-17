package net.faulj.compiler.matrix.affine;

/**
 * Logical ownership of a buffer value.
 */
public enum BufferOwnership {
    BORROWED,
    OWNED,
    NONE
}
