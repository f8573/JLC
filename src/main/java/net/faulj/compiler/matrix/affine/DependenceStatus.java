package net.faulj.compiler.matrix.affine;

/**
 * Tri-state answer for a dependence query.
 */
public enum DependenceStatus {
    PROVEN_DEPENDENCE,
    PROVEN_NONE,
    UNKNOWN
}
