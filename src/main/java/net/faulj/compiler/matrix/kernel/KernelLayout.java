package net.faulj.compiler.matrix.kernel;

/** Storage layout facts visible to portable kernels. */
public enum KernelLayout {
    ROW_MAJOR_DENSE,
    COLUMN_MAJOR_DENSE,
    STRIDED,
    UNKNOWN
}
