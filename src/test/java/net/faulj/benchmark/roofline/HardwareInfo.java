package net.faulj.benchmark.roofline;

final class HardwareInfo {
    final int cores;
    final int simdLanesDouble;
    final boolean fmaEnabled;
    final double clockGhz;
    final int vectorIssueWidth;
    final double peakFlopsPerSecond;
    final String clockSource;

    HardwareInfo(int cores,
                 int simdLanesDouble,
                 boolean fmaEnabled,
                 double clockGhz,
                 int vectorIssueWidth,
                 double peakFlopsPerSecond,
                 String clockSource) {
        this.cores = cores;
        this.simdLanesDouble = simdLanesDouble;
        this.fmaEnabled = fmaEnabled;
        this.clockGhz = clockGhz;
        this.vectorIssueWidth = vectorIssueWidth;
        this.peakFlopsPerSecond = peakFlopsPerSecond;
        this.clockSource = clockSource;
    }
}
