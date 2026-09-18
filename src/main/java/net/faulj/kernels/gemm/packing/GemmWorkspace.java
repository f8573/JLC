package net.faulj.kernels.gemm.packing;

/**
 * Workspace facade for panel packing buffers.
 */
public final class GemmWorkspace {
    private final net.faulj.compute.GemmWorkspace delegate;

    private GemmWorkspace(net.faulj.compute.GemmWorkspace delegate) {
        this.delegate = delegate;
    }

    public static GemmWorkspace get() {
        return new GemmWorkspace(net.faulj.compute.GemmWorkspace.get());
    }

    public double[] getPackA(int requiredSize) {
        return delegate.getPackA(requiredSize);
    }

    public double[] getPackB(int requiredSize) {
        return delegate.getPackB(requiredSize);
    }

    public double[] getTempBuffer(int requiredSize) {
        return delegate.getTempBuffer(requiredSize);
    }

    public void releaseCuda() {
        delegate.releaseCuda();
    }
}
