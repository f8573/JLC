package net.faulj.compiler.matrix.kernel;

/** Observable effect class of one portable kernel operation. */
public enum KernelEffect {
    PURE,
    READS_BUFFER,
    WRITES_BUFFER
}
