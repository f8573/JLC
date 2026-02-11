package net.faulj.benchmark.roofline;

/**
 * Describes the computational profile of a kernel invocation: FLOP count,
 * memory traffic model, and working-set size.
 *
 * <p>For GEMM, the traffic model switches between cold-start (unblocked)
 * and BLIS-style blocked depending on whether blocking parameters are
 * provided.</p>
 */
final class KernelProfile {
    final String kernel;
    final int n;
    final int m;
    final int k;
    final double theoreticalFlops;
    final double actualFlops;
    final double estimatedBytesMoved;
    final double workingSetBytes;

    /** AI model used: "blis-blocked" or "cold-start". */
    final String trafficModel;

    /** Blocking parameters (0 if unblocked / unknown). */
    final int mc;
    final int nc;
    final int kc;

    KernelProfile(String kernel,
                  int n,
                  double theoreticalFlops,
                  double actualFlops,
                  double estimatedBytesMoved,
                  double workingSetBytes) {
        this(kernel, n, n, n, theoreticalFlops, actualFlops, estimatedBytesMoved, workingSetBytes,
            "cold-start", 0, 0, 0);
    }

    KernelProfile(String kernel,
                  int m, int n, int k,
                  double theoreticalFlops,
                  double actualFlops,
                  double estimatedBytesMoved,
                  double workingSetBytes,
                  String trafficModel,
                  int mc, int nc, int kc) {
        this.kernel = kernel;
        this.m = m;
        this.n = n;
        this.k = k;
        this.theoreticalFlops = theoreticalFlops;
        this.actualFlops = actualFlops;
        this.estimatedBytesMoved = estimatedBytesMoved;
        this.workingSetBytes = workingSetBytes;
        this.trafficModel = trafficModel;
        this.mc = mc;
        this.nc = nc;
        this.kc = kc;
    }
}
