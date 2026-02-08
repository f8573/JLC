package net.faulj.decomposition.qr.caqr;

/**
 * Holds final WY aggregation buffers offsets for the panel.
 */
public final class WYBuffers {
    public final int yCombinedOffset;
    public final int tCombinedOffset;

    public WYBuffers(int yCombinedOffset, int tCombinedOffset) {
        this.yCombinedOffset = yCombinedOffset;
        this.tCombinedOffset = tCombinedOffset;
    }
}
