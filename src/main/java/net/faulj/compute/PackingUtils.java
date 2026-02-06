package net.faulj.compute;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Matrix packing utilities for GEMM microkernel.
 * Packs panels of A and B into contiguous buffers with padding to SIMD width.
 */
public final class PackingUtils {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private PackingUtils() {}

    /**
     * Round up value to nearest multiple.
     */
    public static int roundUp(int value, int multiple) {
        if (multiple <= 0) return value;
        int rem = value % multiple;
        return rem == 0 ? value : value + multiple - rem;
    }

    /**
     * Pack A panel: rows × kBlock from row-major A into contiguous buffer.
     * Applies alpha scaling during pack.
     *
     * @param a source matrix data (row-major)
     * @param lda leading dimension of A (stride between rows)
     * @param rowStart starting row index
     * @param rows number of rows to pack
     * @param colStart starting column index (K dimension)
     * @param kBlock number of columns to pack
     * @param alpha scalar to multiply A values
     * @param aPack destination buffer [rows × kBlock]
     */
    public static void packA(double[] a, int lda, int rowStart, int rows,
                            int colStart, int kBlock, double alpha, double[] aPack) {
        int dst = 0;
        if (alpha == 1.0) {
            // Fast path: no scaling
            for (int r = 0; r < rows; r++) {
                int src = (rowStart + r) * lda + colStart;
                System.arraycopy(a, src, aPack, dst, kBlock);
                dst += kBlock;
            }
        } else {
            // Scale during pack
            for (int r = 0; r < rows; r++) {
                int src = (rowStart + r) * lda + colStart;
                for (int k = 0; k < kBlock; k++) {
                    aPack[dst++] = a[src + k] * alpha;
                }
            }
        }
    }

    /**
     * Pack A panel from column-major source.
     */
    public static void packAColMajor(double[] a, int lda, int rowStart, int rows,
                                     int colStart, int kBlock, double alpha, double[] aPack) {
        int dst = 0;
        if (alpha == 1.0) {
            for (int r = 0; r < rows; r++) {
                int row = rowStart + r;
                for (int k = 0; k < kBlock; k++) {
                    int src = (colStart + k) * lda + row;
                    aPack[dst++] = a[src];
                }
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int row = rowStart + r;
                for (int k = 0; k < kBlock; k++) {
                    int src = (colStart + k) * lda + row;
                    aPack[dst++] = a[src] * alpha;
                }
            }
        }
    }

    /**
     * Pack A^T panel (transpose during pack).
     */
    public static void packATranspose(double[] a, int lda, int rowStart, int rows,
                                     int colStart, int kBlock, double alpha, double[] aPack) {
        int dst = 0;
        if (alpha == 1.0) {
            for (int r = 0; r < rows; r++) {
                int col = rowStart + r;
                for (int k = 0; k < kBlock; k++) {
                    int src = (colStart + k) * lda + col;
                    aPack[dst++] = a[src];
                }
            }
        } else {
            for (int r = 0; r < rows; r++) {
                int col = rowStart + r;
                for (int k = 0; k < kBlock; k++) {
                    int src = (colStart + k) * lda + col;
                    aPack[dst++] = a[src] * alpha;
                }
            }
        }
    }

    /**
     * Pack B panel: kBlock × cols from row-major B into contiguous buffer with padding.
     * Pads each row to packedCols (multiple of vecLen) for aligned SIMD access.
     *
     * @param b source matrix data (row-major)
     * @param ldb leading dimension of B
     * @param rowStart starting row index (K dimension)
     * @param kBlock number of rows to pack
     * @param colStart starting column index
     * @param cols actual number of columns to pack
     * @param packedCols padded column count (multiple of vecLen)
     * @param bPack destination buffer [kBlock × packedCols]
     */
    public static void packB(double[] b, int ldb, int rowStart, int kBlock,
                            int colStart, int cols, int packedCols, double[] bPack) {
        int dst = 0;
        for (int k = 0; k < kBlock; k++) {
            int src = (rowStart + k) * ldb + colStart;
            System.arraycopy(b, src, bPack, dst, cols);
            // Zero padding
            if (packedCols > cols) {
                java.util.Arrays.fill(bPack, dst + cols, dst + packedCols, 0.0);
            }
            dst += packedCols;
        }
    }

    /**
     * Pack B panel from column-major source.
     */
    public static void packBColMajor(double[] b, int ldb, int rowStart, int kBlock,
                                     int colStart, int cols, int packedCols, double[] bPack) {
        int dst = 0;
        for (int k = 0; k < kBlock; k++) {
            int row = rowStart + k;
            for (int c = 0; c < cols; c++) {
                int src = (colStart + c) * ldb + row;
                bPack[dst++] = b[src];
            }
            // Zero padding
            if (packedCols > cols) {
                java.util.Arrays.fill(bPack, dst, dst + (packedCols - cols), 0.0);
                dst += (packedCols - cols);
            }
        }
    }

    /**
     * Apply beta scaling to C matrix panel.
     * If beta == 0, zeros C. If beta == 1, no-op.
     */
    public static void scaleCPanel(double[] c, int offset, int rows, int cols,
                                   int ldc, double beta) {
        if (beta == 0.0) {
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * ldc;
                java.util.Arrays.fill(c, rowStart, rowStart + cols, 0.0);
            }
        } else if (beta != 1.0) {
            int vecLen = SPECIES.length();
            for (int r = 0; r < rows; r++) {
                int rowStart = offset + r * ldc;
                int i = 0;
                int loopBound = SPECIES.loopBound(cols);
                DoubleVector betaVec = DoubleVector.broadcast(SPECIES, beta);

                for (; i < loopBound; i += vecLen) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, c, rowStart + i);
                    v.mul(betaVec).intoArray(c, rowStart + i);
                }
                for (; i < cols; i++) {
                    c[rowStart + i] *= beta;
                }
            }
        }
    }
}
