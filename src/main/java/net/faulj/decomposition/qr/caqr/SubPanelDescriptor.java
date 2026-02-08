package net.faulj.decomposition.qr.caqr;

/**
 * Descriptor for a subpanel (leaf) produced by partitioning a full panel.
 */
public final class SubPanelDescriptor {
    public final int rowStart;
    public final int rows;
    public final int cols;
    public final int aOffset;
    public final int yOffset;
    public final int tOffset;
    public final int rOffset;
    public final boolean isShortLeaf;

    public SubPanelDescriptor(int rowStart, int rows, int cols,
                              int aOffset, int yOffset, int tOffset, int rOffset,
                              boolean isShortLeaf) {
        this.rowStart = rowStart;
        this.rows = rows;
        this.cols = cols;
        this.aOffset = aOffset;
        this.yOffset = yOffset;
        this.tOffset = tOffset;
        this.rOffset = rOffset;
        this.isShortLeaf = isShortLeaf;
    }
}
