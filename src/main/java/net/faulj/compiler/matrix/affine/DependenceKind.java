package net.faulj.compiler.matrix.affine;

/**
 * Memory-dependence classification.
 */
public enum DependenceKind {
    RAW,
    WAR,
    WAW,
    REDUCTION
}
