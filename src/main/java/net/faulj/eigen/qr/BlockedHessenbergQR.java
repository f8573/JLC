package net.faulj.eigen.qr;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.matrix.Matrix;

/**
 * Compatibility facade for the canonical Hessenberg reduction engine.
 */
public class BlockedHessenbergQR {
    private static final int DEFAULT_BLOCK_SIZE = 32;
    private static final String CANONICAL_BLOCK_SIZE_PROPERTY = "net.faulj.decomposition.hessenberg.blockSize";
    private static final String LEGACY_BLOCK_SIZE_PROPERTY = "net.faulj.eigen.qr.BlockedHessenbergQR.blockSize";
    private static final String LEGACY_ALIAS_BLOCK_SIZE_PROPERTY = "net.faulj.eigen.qr.blockSize";

    /**
     * Set the panel/block size used by the canonical Hessenberg engine.
     * Must be >= 1. This is primarily intended for benchmarking and testing.
     */
    public static void setBlockSize(int bs) {
        if (bs < 1) {
            throw new IllegalArgumentException("blockSize must be >= 1");
        }
        String value = Integer.toString(bs);
        System.setProperty(CANONICAL_BLOCK_SIZE_PROPERTY, value);
        System.setProperty(LEGACY_BLOCK_SIZE_PROPERTY, value);
        System.setProperty(LEGACY_ALIAS_BLOCK_SIZE_PROPERTY, value);
    }

    /**
     * Return the current block size.
     */
    public static int getBlockSize() {
        return firstPositiveIntegerProperty(
            CANONICAL_BLOCK_SIZE_PROPERTY,
            LEGACY_BLOCK_SIZE_PROPERTY,
            LEGACY_ALIAS_BLOCK_SIZE_PROPERTY
        );
    }

    /**
     * Reduces the matrix A to Hessenberg form.
     *
     * @param A The matrix to reduce.
     * @return The HessenbergResult (H, Q).
     */
    public static HessenbergResult decompose(Matrix A) {
        return HessenbergReduction.decompose(A);
    }

    private static int firstPositiveIntegerProperty(String... keys) {
        for (String key : keys) {
            String value = System.getProperty(key);
            if (value == null || value.isBlank()) {
                continue;
            }
            try {
                int parsed = Integer.parseInt(value.trim());
                if (parsed >= 1) {
                    return parsed;
                }
            } catch (NumberFormatException ignored) {
                // Fall through to the next property key.
            }
        }
        return DEFAULT_BLOCK_SIZE;
    }
}
