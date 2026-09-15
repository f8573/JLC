package net.faulj.compiler.matrix.affine;

/**
 * Semantic operation represented by an affine statement.
 */
public enum StatementKind {
    ADD,
    SCALE,
    TRANSPOSE,
    MATMUL_INIT,
    MATMUL_UPDATE,
    SYNTHETIC
}
