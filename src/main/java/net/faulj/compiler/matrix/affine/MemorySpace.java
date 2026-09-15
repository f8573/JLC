package net.faulj.compiler.matrix.affine;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * Conservative memory-placement classification.
 */
public enum MemorySpace {
    HEAP,
    OFF_HEAP,
    UNKNOWN;

    /**
     * Classify a runtime matrix using only placement facts guaranteed by its
     * public Java type.
     */
    public static MemorySpace from(Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        return matrix instanceof OffHeapMatrix ? OFF_HEAP : HEAP;
    }
}
