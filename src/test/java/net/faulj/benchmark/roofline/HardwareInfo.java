package net.faulj.benchmark.roofline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable snapshot of detected (or assumed) hardware capabilities.
 *
 * <p>Every field that could not be measured carries an explicit assumption
 * annotation in {@link #assumptions}.  The {@link #tier} field records which
 * capability tier was achieved during probing.</p>
 */
final class HardwareInfo {
    final int cores;
    final int simdLanesDouble;
    final boolean fmaEnabled;
    final double clockGhz;
    final int vectorIssueWidth;
    final double peakFlopsPerSecond;
    final String clockSource;
    final CapabilityTier tier;

    /** Human-readable list of assumptions made when probes failed. */
    final List<String> assumptions;

    HardwareInfo(int cores,
                 int simdLanesDouble,
                 boolean fmaEnabled,
                 double clockGhz,
                 int vectorIssueWidth,
                 double peakFlopsPerSecond,
                 String clockSource,
                 CapabilityTier tier,
                 List<String> assumptions) {
        this.cores = cores;
        this.simdLanesDouble = simdLanesDouble;
        this.fmaEnabled = fmaEnabled;
        this.clockGhz = clockGhz;
        this.vectorIssueWidth = vectorIssueWidth;
        this.peakFlopsPerSecond = peakFlopsPerSecond;
        this.clockSource = clockSource;
        this.tier = tier;
        this.assumptions = assumptions == null
            ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(assumptions));
    }
}
