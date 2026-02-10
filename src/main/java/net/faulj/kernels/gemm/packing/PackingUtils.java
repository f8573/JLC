package net.faulj.kernels.gemm.packing;

/**
 * Packing facade used by GEMM kernels.
 */
public final class PackingUtils {
    private PackingUtils() {
    }

    public static int roundUp(int value, int multiple) {
        return net.faulj.compute.PackingUtils.roundUp(value, multiple);
    }

    public static void packA(double[] a, int lda, int rowStart, int rows,
                             int kStart, int kBlock, double alpha, double[] aPack) {
        net.faulj.compute.PackingUtils.packA(a, lda, rowStart, rows, kStart, kBlock, alpha, aPack);
    }

    public static void packATranspose(double[] a, int lda, int rowStart, int rows,
                                      int kStart, int kBlock, double alpha, double[] aPack) {
        net.faulj.compute.PackingUtils.packATranspose(a, lda, rowStart, rows, kStart, kBlock, alpha, aPack);
    }

    public static void packB(double[] b, int ldb, int rowStart, int kBlock,
                             int colStart, int cols, int packedCols, double[] bPack) {
        net.faulj.compute.PackingUtils.packB(b, ldb, rowStart, kBlock, colStart, cols, packedCols, bPack);
    }

    public static void scaleCPanel(double[] c, int cOffset, int rows, int cols, int ldc, double beta) {
        net.faulj.compute.PackingUtils.scaleCPanel(c, cOffset, rows, cols, ldc, beta);
    }
}
