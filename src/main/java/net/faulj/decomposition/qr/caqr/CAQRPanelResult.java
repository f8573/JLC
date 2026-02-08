package net.faulj.decomposition.qr.caqr;

/**
 * Panel-level result container for CAQR (internal use).
 */
public final class CAQRPanelResult {
    public final double[] rTop;
    public final int yCombinedOffset;
    public final int tCombinedOffset;
    public final WorkspaceManager workspace;

    public CAQRPanelResult(double[] rTop, int yCombinedOffset, int tCombinedOffset, WorkspaceManager workspace) {
        this.rTop = rTop;
        this.yCombinedOffset = yCombinedOffset;
        this.tCombinedOffset = tCombinedOffset;
        this.workspace = workspace;
    }
}
