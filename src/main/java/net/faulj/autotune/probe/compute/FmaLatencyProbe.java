package net.faulj.autotune.probe.compute;

import net.faulj.autotune.probe.ProbeConfidence;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Measures FMA pipeline latency by running a single dependent
 * vector-FMA chain and converting throughput to cycles.
 */
public final class FmaLatencyProbe {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private static final int WARMUP = 5;
    private static final int RUNS = 7;
    private static final double FALLBACK_CLOCK_GHZ = 2.5;

    private static volatile double sink;

    private FmaLatencyProbe() {}

    public static FmaLatencyResult run() {
        try {
            return doProbe();
        } catch (Throwable t) {
            return new FmaLatencyResult(4, 0.0, FALLBACK_CLOCK_GHZ,
                    "fallback-default", ProbeConfidence.FAILED);
        }
    }

    private static FmaLatencyResult doProbe() {
        // Calibrate iterations for ≥ 50ms per trial.
        int iterations = calibrate();

        // Warmup.
        for (int w = 0; w < WARMUP; w++) {
            sink += dependentChain(iterations / 10);
        }

        // Measure: best-of-N nanoseconds.
        long bestNs = Long.MAX_VALUE;
        for (int r = 0; r < RUNS; r++) {
            long start = System.nanoTime();
            sink += dependentChain(iterations);
            long elapsed = System.nanoTime() - start;
            bestNs = Math.min(bestNs, elapsed);
        }

        if (bestNs <= 0) {
            return new FmaLatencyResult(4, 0.0, FALLBACK_CLOCK_GHZ,
                    "fallback-default", ProbeConfidence.FAILED);
        }

        double avgNsPerFma = (double) bestNs / iterations;

        // Clock detection.
        ClockEstimate clock = detectClockGhz();

        // latency_cycles = clock_GHz × avgNsPerFma
        double latencyCyclesRaw = clock.ghz * avgNsPerFma;
        int latencyCycles = Math.max(1, (int) Math.round(latencyCyclesRaw));

        // Clamp to sane range [2, 12].
        latencyCycles = Math.max(2, Math.min(12, latencyCycles));

        ProbeConfidence confidence = clock.measured ? ProbeConfidence.MEASURED : ProbeConfidence.ESTIMATED;

        return new FmaLatencyResult(latencyCycles, avgNsPerFma, clock.ghz, clock.source, confidence);
    }

    private static int calibrate() {
        int trial = 1_000_000;
        long start = System.nanoTime();
        sink += dependentChain(trial);
        long elapsed = System.nanoTime() - start;
        double targetNs = 50_000_000.0;
        if (elapsed <= 0) return trial * 10;
        return Math.max(trial, (int) (trial * targetNs / elapsed));
    }

    private static double dependentChain(int iterations) {
        DoubleVector acc = DoubleVector.broadcast(SPECIES, 1.0);
        DoubleVector x = DoubleVector.broadcast(SPECIES, 1.0000001);
        DoubleVector y = DoubleVector.broadcast(SPECIES, 0.9999999);

        for (int i = 0; i < iterations; i++) {
            acc = acc.lanewise(VectorOperators.FMA, x, y);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static ClockEstimate detectClockGhz() {
        Double byProp = parsePositiveDouble(System.getProperty("jlc.roofline.cpu_ghz"));
        if (byProp != null) {
            return new ClockEstimate(byProp, "property:jlc.roofline.cpu_ghz", true);
        }
        Double byEnv = parsePositiveDouble(System.getenv("JLC_ROOFLINE_CPU_GHZ"));
        if (byEnv != null) {
            return new ClockEstimate(byEnv, "env:JLC_ROOFLINE_CPU_GHZ", true);
        }

        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            if (os.contains("win")) {
                String mhz = execFirstLine("powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty MaxClockSpeed)");
                Double val = parsePositiveDouble(mhz);
                if (val != null) return new ClockEstimate(val / 1000.0, "windows-cim", true);
            } else if (os.contains("linux")) {
                String mhz = execFirstLine("bash", "-lc", "awk -F: '/cpu MHz/ {print $2; exit}' /proc/cpuinfo");
                Double val = parsePositiveDouble(mhz);
                if (val != null) return new ClockEstimate(val / 1000.0, "linux-cpuinfo", true);
            } else if (os.contains("mac")) {
                String hz = execFirstLine("sysctl", "-n", "hw.cpufrequency_max");
                Double val = parsePositiveDouble(hz);
                if (val != null) return new ClockEstimate(val / 1e9, "macos-sysctl", true);
            }
        } catch (Exception ignored) {
            // Fall through.
        }

        return new ClockEstimate(FALLBACK_CLOCK_GHZ, "fallback-default", false);
    }

    private static String execFirstLine(String... command) throws Exception {
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            process.waitFor();
            return line == null ? "" : line.trim();
        }
    }

    private static Double parsePositiveDouble(String value) {
        if (value == null) return null;
        try {
            double d = Double.parseDouble(value.trim());
            return d > 0.0 ? d : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static final class ClockEstimate {
        final double ghz;
        final String source;
        final boolean measured;

        ClockEstimate(double ghz, String source, boolean measured) {
            this.ghz = ghz;
            this.source = source;
            this.measured = measured;
        }
    }

    public static final class FmaLatencyResult {
        public final int latencyCycles;
        public final double avgNsPerFma;
        public final double clockGhz;
        public final String clockSource;
        public final ProbeConfidence confidence;

        public FmaLatencyResult(int latencyCycles, double avgNsPerFma,
                                double clockGhz, String clockSource,
                                ProbeConfidence confidence) {
            this.latencyCycles = latencyCycles;
            this.avgNsPerFma = avgNsPerFma;
            this.clockGhz = clockGhz;
            this.clockSource = clockSource;
            this.confidence = confidence;
        }
    }
}
