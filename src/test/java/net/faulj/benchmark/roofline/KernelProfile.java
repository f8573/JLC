package net.faulj.benchmark.roofline;

final class KernelProfile {
    final String kernel;
    final int n;
    final double theoreticalFlops;
    final double actualFlops;
    final double estimatedBytesMoved;
    final double workingSetBytes;

    KernelProfile(String kernel,
                  int n,
                  double theoreticalFlops,
                  double actualFlops,
                  double estimatedBytesMoved,
                  double workingSetBytes) {
        this.kernel = kernel;
        this.n = n;
        this.theoreticalFlops = theoreticalFlops;
        this.actualFlops = actualFlops;
        this.estimatedBytesMoved = estimatedBytesMoved;
        this.workingSetBytes = workingSetBytes;
    }
}
