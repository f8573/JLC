package net.faulj.benchmark.roofline;

final class KernelModel {
    private KernelModel() {
    }

    static KernelProfile gemm(int n) {
        double n3 = (double) n * n * n;
        double flops = 2.0 * n3;
        // Traffic-based estimate for blocked GEMM in Java runtime (includes packing/reloads).
        double bytesMoved = 8.0 * (12.0 * n * (double) n);
        double workingSet = 8.0 * (3.0 * n * (double) n);
        return new KernelProfile("GEMM", n, flops, flops, bytesMoved, workingSet);
    }

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
