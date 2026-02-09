package net.faulj.compute;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import net.faulj.util.PerfTimers;

/**
 * Optimized register-blocked microkernel for GEMM.
 * Implements MR×NR blocking with C kept in registers across K loop.
 *
 * Design:
 * - MR = 6 rows for AVX-512 (8 doubles), 4 rows for AVX2 (4 doubles)
 * - NR = vecLen (SIMD width)
 * - K unrolled by 6-8 to hide FMA latency
 * - Packed A and B with padding to eliminate masks on main path
 */
public final class MicroKernel {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int PREFETCH_DISTANCE = 3;

    private MicroKernel() {}

    /**
     * Select optimal MR (micro rows) based on SIMD width.
     * Goal: MR × NR = 16-48 elements in registers (fits in register file).
     * For AVX2 (16 YMM registers): 4 accumulators + temps = safe choice.
     */
    public static int optimalMR(int vecLen) {
        if (vecLen >= 8) {
            return 6;  // 6×8 = 48 doubles in registers (AVX-512)
        }
        if (vecLen >= 4) {
            return 4;  // 4×4 = 16 doubles in registers (AVX2)
        }
        return 2;
    }

    /**
     * Select optimal K unroll factor.
     * Goal: Hide FMA latency (4-5 cycles) with independent operations.
     * Using 8 provides good ILP and hides FMA latency effectively.
     */
    public static int optimalKUnroll(int vecLen) {
        // Allow runtime override
        try {
            String ku = System.getProperty("la.gemm.kunroll");
            if (ku != null) {
                int kuv = Integer.parseInt(ku);
                if (kuv > 0) return kuv;
            }
        } catch (Exception ignored) {
        }
        // Tuned default: use k-unroll=8 to hide FMA latency on this hardware
        return 8;
    }

    /**
     * Core microkernel: compute C[mr, nr] += A[mr, k] * B[k, nr].
     *
     * @param mr actual rows to compute (1-6)
     * @param kBlock K dimension of this panel
     * @param packedN padded N dimension (multiple of vecLen)
     * @param actualN actual N dimension (for bounds check)
     * @param aPack packed A panel [mr × kBlock], row-major, scaled by alpha
     * @param bPack packed B panel [kBlock × packedN], row-major, padded
     * @param c output matrix C
     * @param cOffset base offset into C
     * @param ldc leading dimension of C
     */
    public static void compute(int mr, int kBlock, int packedN, int actualN,
                               double[] aPack, double[] bPack,
                               double[] c, int cOffset, int ldc) {
        int vecLen = SPECIES.length();
        int kUnroll = optimalKUnroll(vecLen);

        boolean aligned = (actualN % vecLen) == 0;
        int jLimit = aligned ? actualN : actualN - (actualN % vecLen);

        // Precompute A panel offsets for each row
        int[] aOffsets = new int[6];
        for (int r = 1; r < mr && r < 6; r++) {
            aOffsets[r] = r * kBlock;
        }

        // Main aligned loop over NR blocks
        for (int j = 0; j < jLimit; j += vecLen) {
            computeBlock(mr, kBlock, packedN, j, aPack, aOffsets, bPack,
                        c, cOffset, ldc, vecLen, kUnroll, false, null);
        }

        // Tail with mask if not aligned
        if (!aligned) {
            VectorMask<Double> mask = SPECIES.indexInRange(jLimit, actualN);
            computeBlock(mr, kBlock, packedN, jLimit, aPack, aOffsets, bPack,
                        c, cOffset, ldc, vecLen, kUnroll, true, mask);
        }
    }

    /**
     * Compute one NR block of the microkernel.
     * Uses scalar variables instead of arrays to avoid allocations in inner loop.
     */
    private static void computeBlock(int mr, int kBlock, int packedN, int j,
                                     double[] aPack, int[] aOffsets, double[] bPack,
                                     double[] c, int cOffset, int ldc,
                                     int vecLen, int kUnroll,
                                     boolean useMask, VectorMask<Double> mask) {
        long tMK = PerfTimers.start();
        // Load C registers
        int cBase = cOffset + j;
        DoubleVector c0, c1, c2, c3, c4, c5;

        if (useMask) {
            c0 = DoubleVector.fromArray(SPECIES, c, cBase, mask);
            c1 = mr > 1 ? DoubleVector.fromArray(SPECIES, c, cBase + ldc, mask) : DoubleVector.zero(SPECIES);
            c2 = mr > 2 ? DoubleVector.fromArray(SPECIES, c, cBase + 2 * ldc, mask) : DoubleVector.zero(SPECIES);
            c3 = mr > 3 ? DoubleVector.fromArray(SPECIES, c, cBase + 3 * ldc, mask) : DoubleVector.zero(SPECIES);
            c4 = mr > 4 ? DoubleVector.fromArray(SPECIES, c, cBase + 4 * ldc, mask) : DoubleVector.zero(SPECIES);
            c5 = mr > 5 ? DoubleVector.fromArray(SPECIES, c, cBase + 5 * ldc, mask) : DoubleVector.zero(SPECIES);
        } else {
            c0 = DoubleVector.fromArray(SPECIES, c, cBase);
            c1 = mr > 1 ? DoubleVector.fromArray(SPECIES, c, cBase + ldc) : DoubleVector.zero(SPECIES);
            c2 = mr > 2 ? DoubleVector.fromArray(SPECIES, c, cBase + 2 * ldc) : DoubleVector.zero(SPECIES);
            c3 = mr > 3 ? DoubleVector.fromArray(SPECIES, c, cBase + 3 * ldc) : DoubleVector.zero(SPECIES);
            c4 = mr > 4 ? DoubleVector.fromArray(SPECIES, c, cBase + 4 * ldc) : DoubleVector.zero(SPECIES);
            c5 = mr > 5 ? DoubleVector.fromArray(SPECIES, c, cBase + 5 * ldc) : DoubleVector.zero(SPECIES);
        }

        // K loop with unrolling by 8 (match optimalKUnroll for FMA latency hiding)
        int p = 0;
        int kLimit = kBlock - 7;

        // Unrolled K loop by 8 - no array allocations!
        for (; p < kLimit; p += 8) {
            int bBase = p * packedN + j;

            // Load B vectors (8 unrolled)
            DoubleVector b0 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase);
            DoubleVector b1 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + packedN);
            DoubleVector b2 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 2 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 2 * packedN);
            DoubleVector b3 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 3 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 3 * packedN);
            DoubleVector b4 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 4 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 4 * packedN);
            DoubleVector b5 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 5 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 5 * packedN);
            DoubleVector b6 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 6 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 6 * packedN);
            DoubleVector b7 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, bBase + 7 * packedN, mask)
                : DoubleVector.fromArray(SPECIES, bPack, bBase + 7 * packedN);

            // Row 0 - 8 FMAs
            c0 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p]), c0);
            c0 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 1]), c0);
            c0 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 2]), c0);
            c0 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 3]), c0);
            c0 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 4]), c0);
            c0 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 5]), c0);
            c0 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 6]), c0);
            c0 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[p + 7]), c0);

            if (mr > 1) {
                int off1 = aOffsets[1];
                c1 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p]), c1);
                c1 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 1]), c1);
                c1 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 2]), c1);
                c1 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 3]), c1);
                c1 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 4]), c1);
                c1 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 5]), c1);
                c1 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 6]), c1);
                c1 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off1 + p + 7]), c1);
            }

            if (mr > 2) {
                int off2 = aOffsets[2];
                c2 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p]), c2);
                c2 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 1]), c2);
                c2 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 2]), c2);
                c2 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 3]), c2);
                c2 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 4]), c2);
                c2 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 5]), c2);
                c2 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 6]), c2);
                c2 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off2 + p + 7]), c2);
            }

            if (mr > 3) {
                int off3 = aOffsets[3];
                c3 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p]), c3);
                c3 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 1]), c3);
                c3 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 2]), c3);
                c3 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 3]), c3);
                c3 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 4]), c3);
                c3 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 5]), c3);
                c3 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 6]), c3);
                c3 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off3 + p + 7]), c3);
            }

            if (mr > 4) {
                int off4 = aOffsets[4];
                c4 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p]), c4);
                c4 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 1]), c4);
                c4 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 2]), c4);
                c4 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 3]), c4);
                c4 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 4]), c4);
                c4 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 5]), c4);
                c4 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 6]), c4);
                c4 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off4 + p + 7]), c4);
            }

            if (mr > 5) {
                int off5 = aOffsets[5];
                c5 = b0.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p]), c5);
                c5 = b1.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 1]), c5);
                c5 = b2.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 2]), c5);
                c5 = b3.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 3]), c5);
                c5 = b4.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 4]), c5);
                c5 = b5.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 5]), c5);
                c5 = b6.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 6]), c5);
                c5 = b7.lanewise(VectorOperators.FMA, DoubleVector.broadcast(SPECIES, aPack[off5 + p + 7]), c5);
            }
        }

        // Remainder K loop (no unroll)
        for (; p < kBlock; p++) {
            DoubleVector b0 = useMask
                ? DoubleVector.fromArray(SPECIES, bPack, p * packedN + j, mask)
                : DoubleVector.fromArray(SPECIES, bPack, p * packedN + j);

            DoubleVector a0v = DoubleVector.broadcast(SPECIES, aPack[p]);
            c0 = b0.lanewise(VectorOperators.FMA, a0v, c0);

            if (mr > 1) {
                DoubleVector a1v = DoubleVector.broadcast(SPECIES, aPack[aOffsets[1] + p]);
                c1 = b0.lanewise(VectorOperators.FMA, a1v, c1);
            }
            if (mr > 2) {
                DoubleVector a2v = DoubleVector.broadcast(SPECIES, aPack[aOffsets[2] + p]);
                c2 = b0.lanewise(VectorOperators.FMA, a2v, c2);
            }
            if (mr > 3) {
                DoubleVector a3v = DoubleVector.broadcast(SPECIES, aPack[aOffsets[3] + p]);
                c3 = b0.lanewise(VectorOperators.FMA, a3v, c3);
            }
            if (mr > 4) {
                DoubleVector a4v = DoubleVector.broadcast(SPECIES, aPack[aOffsets[4] + p]);
                c4 = b0.lanewise(VectorOperators.FMA, a4v, c4);
            }
            if (mr > 5) {
                DoubleVector a5v = DoubleVector.broadcast(SPECIES, aPack[aOffsets[5] + p]);
                c5 = b0.lanewise(VectorOperators.FMA, a5v, c5);
            }
        }

        // Store C registers
        if (useMask) {
            c0.intoArray(c, cBase, mask);
            if (mr > 1) c1.intoArray(c, cBase + ldc, mask);
            if (mr > 2) c2.intoArray(c, cBase + 2 * ldc, mask);
            if (mr > 3) c3.intoArray(c, cBase + 3 * ldc, mask);
            if (mr > 4) c4.intoArray(c, cBase + 4 * ldc, mask);
            if (mr > 5) c5.intoArray(c, cBase + 5 * ldc, mask);
        } else {
            c0.intoArray(c, cBase);
            if (mr > 1) c1.intoArray(c, cBase + ldc);
            if (mr > 2) c2.intoArray(c, cBase + 2 * ldc);
            if (mr > 3) c3.intoArray(c, cBase + 3 * ldc);
            if (mr > 4) c4.intoArray(c, cBase + 4 * ldc);
            if (mr > 5) c5.intoArray(c, cBase + 5 * ldc);
        }
        PerfTimers.record("MicroKernel.compute", tMK);
    }

    /**
     * Software prefetch hint (may be optimized away by JIT).
     */
    private static void prefetchB(double[] bPack, int index) {
        // Touch the data to hint prefetch
        if (index < bPack.length) {
            double unused = bPack[index];
        }
    }
}
