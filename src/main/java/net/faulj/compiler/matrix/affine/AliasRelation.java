package net.faulj.compiler.matrix.affine;

/**
 * Conservative relation between two logical buffers.
 */
public enum AliasRelation {
    MUST_ALIAS,
    NO_ALIAS,
    MAY_ALIAS
}
