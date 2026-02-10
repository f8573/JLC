package net.faulj.kernels.gemm.dispatch;

/**
 * Public dispatch facade for GEMM kernel selection and block-size heuristics.
 */
public final class GemmDispatch {
    private GemmDispatch() {
    }

    public enum Kernel {
        TINY,
        SMALL,
        MATVEC,
        MICROKERNEL,
        CUDA,
        PARALLEL_MICRO
    }

    public static final class BlockSizes {
        public final int mc;
        public final int kc;
        public final int nc;
        public final int mr;
        public final int nr;

        public BlockSizes(int mc, int kc, int nc, int mr, int nr) {
            this.mc = mc;
            this.kc = kc;
            this.nc = nc;
            this.mr = mr;
            this.nr = nr;
        }

        @Override
        public String toString() {
            return String.format("MC=%d, KC=%d, NC=%d, MR=%d, NR=%d", mc, kc, nc, mr, nr);
        }
    }

    public static Kernel selectKernel(int m, int n, int k, boolean cudaAvailable, int threads) {
        net.faulj.compute.GemmDispatch.Kernel selected =
            net.faulj.compute.GemmDispatch.selectKernel(m, n, k, cudaAvailable, threads);
        return switch (selected) {
            case TINY -> Kernel.TINY;
            case SMALL -> Kernel.SMALL;
            case MATVEC -> Kernel.MATVEC;
            case MICROKERNEL -> Kernel.MICROKERNEL;
            case CUDA -> Kernel.CUDA;
            case PARALLEL_MICRO -> Kernel.PARALLEL_MICRO;
        };
    }

    public static BlockSizes computeBlockSizes() {
        net.faulj.compute.GemmDispatch.BlockSizes bs = net.faulj.compute.GemmDispatch.computeBlockSizes();
        return new BlockSizes(bs.mc, bs.kc, bs.nc, bs.mr, bs.nr);
    }

    public static int optimalParallelism(int m, int n, int k, int maxThreads, BlockSizes blocks) {
        net.faulj.compute.GemmDispatch.BlockSizes delegate =
            new net.faulj.compute.GemmDispatch.BlockSizes(blocks.mc, blocks.kc, blocks.nc, blocks.mr, blocks.nr);
        return net.faulj.compute.GemmDispatch.optimalParallelism(m, n, k, maxThreads, delegate);
    }

    public static boolean isEffectivelySparse(double[] data, int sampleSize) {
        return net.faulj.compute.GemmDispatch.isEffectivelySparse(data, sampleSize);
    }
}
