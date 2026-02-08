package net.faulj.decomposition.qr.caqr;

/**
 * Node in the TSQR reduction tree (internal or leaf).
 */
public final class ReductionNode {
    public final int nodeId;
    public final int level;
    public final int rStackOffset;
    public final int yNodeOffset;
    public final int tNodeOffset;
    public final int rOutOffset;
    public final int leftChild;
    public final int rightChild;

    public ReductionNode(int nodeId, int level, int rStackOffset, int yNodeOffset,
                         int tNodeOffset, int rOutOffset, int leftChild, int rightChild) {
        this.nodeId = nodeId;
        this.level = level;
        this.rStackOffset = rStackOffset;
        this.yNodeOffset = yNodeOffset;
        this.tNodeOffset = tNodeOffset;
        this.rOutOffset = rOutOffset;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
    }
}
