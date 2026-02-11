package net.faulj.benchmark.roofline;

/**
 * First-principles computational models for kernel FLOP counts and memory
 * traffic estimates.
 *
 * <p>For GEMM, two traffic models are supported:
 * <ul>
 *   <li><b>Cold-start</b>: all three matrices loaded from memory once
 *       (C read + written).  Used when matrices fit in cache or blocking
 *       parameters are unknown.</li>
 *   <li><b>BLIS-blocked</b>: traffic model for BLIS/GotoBLAS 5-loop nest
 *       with panels packed into cache.  Requires MC, NC, KC.</li>
 * </ul>
 */
final class KernelModel {
    private KernelModel() {
    }

    // ── GEMM models ────────────────────────────────────────────────────

    /**
     * Square GEMM with cold-start (conservative) traffic model.
     * Use when blocking parameters are not known.
     */
    static KernelProfile gemm(int n) {
        return gemm(n, n, n, 0, 0, 0);
    }

    /**
     * Rectangular GEMM with blocking-aware traffic model.
     *
     * <p>If mc, nc, kc are all positive, uses the BLIS 5-loop traffic model.
     * Otherwise falls back to the cold-start model.</p>
     *
     * @param m  rows of A / rows of C
     * @param n  columns of B / columns of C
     * @param k  columns of A / rows of B
     * @param mc M-blocking parameter (0 = unknown)
     * @param nc N-blocking parameter (0 = unknown)
     * @param kc K-blocking parameter (0 = unknown)
     */
    static KernelProfile gemm(int m, int n, int k, int mc, int nc, int kc) {
        double flops = 2.0 * m * (double) n * k;
        double workingSet = 8.0 * (m * (double) k + k * (double) n + m * (double) n);

        double bytesMoved;
        String model;

        if (mc > 0 && nc > 0 && kc > 0) {
            // BLIS 5-loop traffic model:
            //   B panels: loaded once       → 8 × K × N
            //   A panels: reloaded per jc   → 8 × M × K × ⌈N/NC⌉
            //   C blocks: read+write per pc → 8 × M × N × 2 × ⌈K/KC⌉
            double jcIters = Math.ceil((double) n / nc);
            double pcIters = Math.ceil((double) k / kc);
            double bytesB = 8.0 * k * (double) n;
            double bytesA = 8.0 * m * (double) k * jcIters;
            double bytesC = 8.0 * m * (double) n * 2.0 * pcIters;
            bytesMoved = bytesB + bytesA + bytesC;
            model = "blis-blocked";
        } else {
            // Cold-start model: read A, read B, read+write C.
            bytesMoved = 8.0 * (m * (double) k + k * (double) n + 2.0 * m * (double) n);
            model = "cold-start";
        }

        return new KernelProfile("GEMM", m, n, k, flops, flops, bytesMoved, workingSet,
            model, mc, nc, kc);
    }

    // ── Non-GEMM models (unchanged, cold-start only) ──────────────────

    static KernelProfile qr(int n) {
        double n3 = (double) n * n * n;
        double flops = (4.0 / 3.0) * n3;
        double bytesMoved = 8.0 * (0.056 * n3 + 20.0 * n * (double) n);
        double workingSet = 8.0 * (20.0 * n * (double) n);
        return new KernelProfile("QR", n, flops, flops, bytesMoved, workingSet);
    }

    static KernelProfile lu(int n) {
        double n3 = (double) n * n * n;
        double flops = (2.0 / 3.0) * n3;
        double bytesMoved = 8.0 * (0.035 * n3 + 12.0 * n * (double) n);
        double workingSet = 8.0 * (12.0 * n * (double) n);
        return new KernelProfile("LU", n, flops, flops, bytesMoved, workingSet);
    }

    static KernelProfile hessenberg(int n) {
        double n3 = (double) n * n * n;
        double flops = (10.0 / 3.0) * n3;
        double bytesMoved = 8.0 * (0.14 * n3 + 24.0 * n * (double) n);
        double workingSet = 8.0 * (24.0 * n * (double) n);
        return new KernelProfile("Hessenberg", n, flops, flops, bytesMoved, workingSet);
    }

    static KernelProfile schur(int n) {
        double n3 = (double) n * n * n;
        double flops = 15.0 * n3;
        double bytesMoved = 8.0 * (1.00 * n3 + 30.0 * n * (double) n);
        double workingSet = 8.0 * (30.0 * n * (double) n);
        return new KernelProfile("Schur", n, flops, flops, bytesMoved, workingSet);
    }

    static KernelProfile svd(int n) {
        double n3 = (double) n * n * n;
        double flops = 11.0 * n3;
        double bytesMoved = 8.0 * (0.70 * n3 + 24.0 * n * (double) n);
        double workingSet = 8.0 * (24.0 * n * (double) n);
        return new KernelProfile("SVD", n, flops, flops, bytesMoved, workingSet);
    }
}
