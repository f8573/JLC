package net.faulj.compiler.matrix.affine;

/**
 * Logical category of a value visible to the affine program.
 *
 * <p>These categories describe compiler semantics only. They are not runtime
 * allocation instructions.</p>
 */
public enum BufferKind {
    EXTERNAL_INPUT,
    TEMPORARY,
    SYMBOLIC
}
