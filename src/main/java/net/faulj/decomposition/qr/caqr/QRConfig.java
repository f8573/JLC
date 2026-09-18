package net.faulj.decomposition.qr.caqr;

/**
 * Configuration object for Communication-Avoiding QR (CAQR).
 */
public final class QRConfig {
    public final int b;
    public final int p;
    public final int alignmentBytes;
    public final int numThreads;
    public final double l2TargetFrac;

    public QRConfig(int b, int p, int alignmentBytes, int numThreads, double l2TargetFrac) {
        this.b = b;
        this.p = p;
        this.alignmentBytes = alignmentBytes;
        this.numThreads = numThreads;
        this.l2TargetFrac = l2TargetFrac;
    }

    public static QRConfig defaultConfig() {
        return new QRConfig(64, 0, 64, Runtime.getRuntime().availableProcessors(), 0.6);
    }
}
