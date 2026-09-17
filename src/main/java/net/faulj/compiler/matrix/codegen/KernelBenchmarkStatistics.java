package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Robust timing summary retaining every raw sample used by the decision. */
public final class KernelBenchmarkStatistics {
    private final List<Long> samplesNanos;
    private final double medianNanos;
    private final double minNanos;
    private final double maxNanos;
    private final double madNanos;

    private KernelBenchmarkStatistics(List<Long> samplesNanos,
                                      double medianNanos,
                                      double minNanos,
                                      double maxNanos,
                                      double madNanos) {
        this.samplesNanos = List.copyOf(samplesNanos);
        this.medianNanos = medianNanos;
        this.minNanos = minNanos;
        this.maxNanos = maxNanos;
        this.madNanos = madNanos;
    }

    public static KernelBenchmarkStatistics from(long[] samplesNanos) {
        if (samplesNanos == null || samplesNanos.length == 0) {
            throw new IllegalArgumentException("At least one timing sample is required");
        }
        long[] sorted = samplesNanos.clone();
        for (long sample : sorted) {
            if (sample <= 0L) {
                throw new IllegalArgumentException("Timing samples must be positive");
            }
        }
        Arrays.sort(sorted);
        double median = median(sorted);
        double[] deviations = new double[sorted.length];
        for (int index = 0; index < sorted.length; index++) {
            deviations[index] = Math.abs(sorted[index] - median);
        }
        Arrays.sort(deviations);
        List<Long> retained = new ArrayList<>(sorted.length);
        for (long sample : samplesNanos) {
            retained.add(sample);
        }
        return new KernelBenchmarkStatistics(retained, median, sorted[0], sorted[sorted.length - 1],
            median(deviations));
    }

    public static KernelBenchmarkStatistics from(List<Long> samplesNanos) {
        if (samplesNanos == null) {
            throw new IllegalArgumentException("Samples must not be null");
        }
        long[] values = new long[samplesNanos.size()];
        for (int index = 0; index < values.length; index++) {
            values[index] = samplesNanos.get(index);
        }
        return from(values);
    }

    public List<Long> samplesNanos() {
        return samplesNanos;
    }

    public List<Long> rawSamplesNanos() {
        return samplesNanos;
    }

    public int sampleCount() {
        return samplesNanos.size();
    }

    public int retainedSampleCount() {
        return samplesNanos.size();
    }

    public double medianNanos() {
        return medianNanos;
    }

    public double minNanos() {
        return minNanos;
    }

    public double maxNanos() {
        return maxNanos;
    }

    public double madNanos() {
        return madNanos;
    }

    public double relativeMad() {
        return medianNanos <= 0.0 ? Double.POSITIVE_INFINITY : madNanos / medianNanos;
    }

    public boolean stable(double threshold) {
        return Double.isFinite(threshold) && threshold >= 0.0 && relativeMad() <= threshold;
    }

    public double nsPerElement(long elements) {
        return elements <= 0L ? 0.0 : medianNanos / (double) elements;
    }

    @Override
    public String toString() {
        return "median=" + medianNanos + "ns min=" + minNanos + "ns max=" + maxNanos
            + "ns MAD=" + madNanos + "ns samples=" + sampleCount();
    }

    private static double median(long[] sorted) {
        int middle = sorted.length / 2;
        if ((sorted.length & 1) != 0) {
            return sorted[middle];
        }
        return sorted[middle - 1] / 2.0 + sorted[middle] / 2.0;
    }

    private static double median(double[] sorted) {
        int middle = sorted.length / 2;
        if ((sorted.length & 1) != 0) {
            return sorted[middle];
        }
        return sorted[middle - 1] / 2.0 + sorted[middle] / 2.0;
    }
}
