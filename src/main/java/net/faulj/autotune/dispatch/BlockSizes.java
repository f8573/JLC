package net.faulj.autotune.dispatch;

/**
 * Container for tuned block sizes to be installed into runtime dispatch.
 */
public final class BlockSizes {
    public final int mr;
    public final int nr;
    public final int kc;
    public final int kUnroll;
    public final int mc;
    public final int nc;

    public BlockSizes(int mr, int nr, int kc, int kUnroll, int mc, int nc) {
        this.mr = mr;
        this.nr = nr;
        this.kc = kc;
        this.kUnroll = kUnroll;
        this.mc = mc;
        this.nc = nc;
    }
}
