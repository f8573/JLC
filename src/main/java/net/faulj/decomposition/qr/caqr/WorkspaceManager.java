package net.faulj.decomposition.qr.caqr;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

/**
 * Workspace manager: single aligned allocation per panel.
 * This is a lightweight manager that exposes a DoubleBuffer view.
 */
public final class WorkspaceManager {
    private final ByteBuffer rawBuffer;
    private final DoubleBuffer dbuf;
    private final int totalDoubles;
    private final int alignmentBytes;

    public final int panelAOffset;
    public final int yCombinedOffset;
    public final int tCombinedOffset;

    public WorkspaceManager(int m, int b, int p, int alignmentBytes) {
        this.alignmentBytes = alignmentBytes;
        this.totalDoubles = computeWorkspaceDoubles(m, b, p);
        int totalBytes = totalDoubles * Double.BYTES;
        this.rawBuffer = ByteBuffer.allocateDirect(totalBytes + alignmentBytes);
        int alignedPos = alignBufferOffset(rawBuffer, alignmentBytes);
        ByteBuffer slice = ((ByteBuffer) rawBuffer.position(alignedPos)).slice();
        slice.order(ByteOrder.nativeOrder());
        this.dbuf = slice.asDoubleBuffer();
        this.panelAOffset = 0;
        // simple layout: panelA | yCombined | tCombined (offsets in doubles)
        this.yCombinedOffset = panelAOffset + m * b;
        this.tCombinedOffset = yCombinedOffset + m * b;
    }

    public DoubleBuffer doubleBuffer() { return dbuf; }
    public ByteBuffer byteBuffer() { return rawBuffer; }

    public static int computeWorkspaceDoubles(int m, int b, int p) {
        int panelA = m * b;
        int leaves = p * (b * b); // T_i
        int nodes = Math.max(0, p - 1) * (b * b * 3); // crude estimate for node buffers
        int ycomb = m * b;
        int tcomb = b * b;
        int scratch = b * Math.max(b, 128);
        int total = panelA + leaves + nodes + ycomb + tcomb + scratch + 64;
        return total;
    }

    private static int alignBufferOffset(ByteBuffer bb, int alignment) {
        // attempt to align position to a multiple of alignment bytes (best-effort)
        int pos = 0; // allocateDirect is commonly aligned; return 0 for portability
        return pos;
    }
}
