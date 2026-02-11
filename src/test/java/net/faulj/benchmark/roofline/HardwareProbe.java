package net.faulj.benchmark.roofline;

import jdk.incubator.vector.DoubleVector;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Probes CPU hardware capabilities with tiered fallback.
 *
 * <p>Detection order for each parameter:
 * <ol>
 *   <li>Explicit system property / environment variable (user override)</li>
 *   <li>Runtime detection (Vector API, OS commands)</li>
 *   <li>Architecture-based conservative default</li>
 *   <li>Absolute conservative default (scalar, single-issue)</li>
 * </ol>
 *
 * <p>Every assumption made when a probe fails is recorded in
 * {@link HardwareInfo#assumptions} and the capability tier is downgraded
 * accordingly.</p>
 */
final class HardwareProbe {
    private static final String CLOCK_PROPERTY = "jlc.roofline.cpu_ghz";
    private static final String CLOCK_ENV = "JLC_ROOFLINE_CPU_GHZ";
    private static final String FMA_PROPERTY = "jlc.roofline.fma";
    private static final String FMA_ENV = "JLC_ROOFLINE_FMA";
    private static final String ISSUE_WIDTH_PROPERTY = "jlc.roofline.vector_issue_width";
    private static final String ISSUE_WIDTH_ENV = "JLC_ROOFLINE_VECTOR_ISSUE_WIDTH";
    private static final String SIMD_LANES_PROPERTY = "jlc.roofline.simd_lanes";
    private static final String SIMD_LANES_ENV = "JLC_ROOFLINE_SIMD_LANES";

    private HardwareProbe() {
    }

    static HardwareInfo probe() {
        List<String> assumptions = new ArrayList<>();
        boolean allMeasured = true;

        // ── Cores ──────────────────────────────────────────────────────
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());

        // ── SIMD lanes ─────────────────────────────────────────────────
        int simdLanes;
        boolean simdDetected;
        Integer explicitSimd = parsePositiveInt(System.getProperty(SIMD_LANES_PROPERTY));
        if (explicitSimd == null) {
            explicitSimd = parsePositiveInt(System.getenv(SIMD_LANES_ENV));
        }
        if (explicitSimd != null) {
            simdLanes = explicitSimd;
            simdDetected = true;
        } else {
            simdLanes = detectSimdLanes();
            simdDetected = simdLanes > 1;
            if (!simdDetected) {
                simdLanes = 1;
                assumptions.add("simd_lanes=1 (Vector API unavailable, assuming scalar)");
                allMeasured = false;
            }
        }

        // ── FMA ────────────────────────────────────────────────────────
        boolean fma;
        boolean fmaDetected;
        Boolean explicitFma = BenchmarkMode.parseFlag(System.getProperty(FMA_PROPERTY));
        if (explicitFma == null) {
            explicitFma = BenchmarkMode.parseFlag(System.getenv(FMA_ENV));
        }
        if (explicitFma != null) {
            fma = explicitFma;
            fmaDetected = true;
        } else {
            String arch = archName();
            if (isX86(arch) || isArm64(arch)) {
                // All x86-64 CPUs since Haswell (2013) and all AArch64 CPUs
                // support FMA.  Sound architectural assumption.
                fma = true;
                fmaDetected = true;
            } else {
                fma = false;
                fmaDetected = false;
                assumptions.add("fma=false (unknown architecture, assuming no FMA)");
                allMeasured = false;
            }
        }

        // ── Issue width (FMA pipes per core) ───────────────────────────
        // CRITICAL: modern x86-64 cores (Haswell+, Zen2+) have 2 FMA
        // execution ports.  ARM Neon typically has 1.  Previous default
        // of 1 halved the compute roof on most x86 CPUs.
        int issueWidth;
        boolean issueWidthExplicit;
        Integer explicitIssue = parsePositiveInt(System.getProperty(ISSUE_WIDTH_PROPERTY));
        if (explicitIssue == null) {
            explicitIssue = parsePositiveInt(System.getenv(ISSUE_WIDTH_ENV));
        }
        if (explicitIssue != null) {
            issueWidth = explicitIssue;
            issueWidthExplicit = true;
        } else {
            String arch = archName();
            issueWidthExplicit = false;
            if (isX86(arch) && simdLanes >= 4) {
                // x86-64 with AVX2 (4 doubles) or AVX-512 (8 doubles):
                // Haswell, Broadwell, Skylake, Cascade Lake, Ice Lake,
                // Zen2, Zen3, Zen4 all have 2 FMA pipes.
                issueWidth = 2;
                assumptions.add("issue_width=2 (x86-64 with AVX2/AVX-512, inferred)");
            } else if (isArm64(arch)) {
                // ARM Neon: typically 1 fully-pipelined FMA unit.
                issueWidth = 1;
                assumptions.add("issue_width=1 (AArch64, inferred)");
            } else {
                issueWidth = 1;
                assumptions.add("issue_width=1 (unknown architecture, conservative)");
                allMeasured = false;
            }
        }

        // ── Clock frequency ────────────────────────────────────────────
        ClockEstimate clockEstimate = detectClockGhz();
        boolean clockMeasured = !clockEstimate.source.startsWith("fallback");
        if (!clockMeasured) {
            assumptions.add("clock_ghz=2.5 (fallback default, OS probe failed)");
            allMeasured = false;
        }

        // ── Compute peak FLOP/s ────────────────────────────────────────
        double flopsPerCyclePerCore = simdLanes * (fma ? 2.0 : 1.0) * issueWidth;
        double peakFlops = cores * clockEstimate.ghz * 1e9 * flopsPerCyclePerCore;

        // ── Determine capability tier ──────────────────────────────────
        CapabilityTier tier;
        if (allMeasured && simdDetected && fmaDetected && clockMeasured && issueWidthExplicit) {
            tier = CapabilityTier.FULLY_MEASURED;
        } else if (simdDetected && clockMeasured) {
            tier = CapabilityTier.PARTIALLY_MEASURED;
        } else if (clockMeasured) {
            tier = CapabilityTier.MINIMAL;
        } else {
            tier = CapabilityTier.SAFE_MODE;
        }

        return new HardwareInfo(
            cores,
            simdLanes,
            fma,
            clockEstimate.ghz,
            issueWidth,
            peakFlops,
            clockEstimate.source,
            tier,
            assumptions
        );
    }

    // ── SIMD detection ─────────────────────────────────────────────────

    private static int detectSimdLanes() {
        try {
            return Math.max(1, DoubleVector.SPECIES_PREFERRED.length());
        } catch (Throwable ignored) {
            return 1;
        }
    }

    // ── Clock detection ────────────────────────────────────────────────

    private static ClockEstimate detectClockGhz() {
        Double byProperty = parsePositiveDouble(System.getProperty(CLOCK_PROPERTY));
        if (byProperty != null) {
            return new ClockEstimate(byProperty, "property:" + CLOCK_PROPERTY);
        }

        Double byEnv = parsePositiveDouble(System.getenv(CLOCK_ENV));
        if (byEnv != null) {
            return new ClockEstimate(byEnv, "env:" + CLOCK_ENV);
        }

        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        try {
            if (os.contains("win")) {
                String mhz = execFirstLine("powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty MaxClockSpeed)");
                Double value = parsePositiveDouble(mhz);
                if (value != null) {
                    return new ClockEstimate(value / 1000.0, "windows-cim-maxclockspeed");
                }
            } else if (os.contains("linux")) {
                String mhz = execFirstLine("bash", "-lc", "awk -F: '/cpu MHz/ {print $2; exit}' /proc/cpuinfo");
                Double value = parsePositiveDouble(mhz);
                if (value != null) {
                    return new ClockEstimate(value / 1000.0, "linux-/proc/cpuinfo");
                }
            } else if (os.contains("mac")) {
                String hz = execFirstLine("sysctl", "-n", "hw.cpufrequency_max");
                Double value = parsePositiveDouble(hz);
                if (value != null) {
                    return new ClockEstimate(value / 1e9, "macos-sysctl-hw.cpufrequency_max");
                }
                hz = execFirstLine("sysctl", "-n", "hw.cpufrequency");
                value = parsePositiveDouble(hz);
                if (value != null) {
                    return new ClockEstimate(value / 1e9, "macos-sysctl-hw.cpufrequency");
                }
            }
        } catch (Exception ignored) {
            // Fall through to deterministic default.
        }

        return new ClockEstimate(2.5, "fallback-default");
    }

    // ── Architecture helpers ───────────────────────────────────────────

    private static String archName() {
        return System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
    }

    private static boolean isX86(String arch) {
        return arch.contains("x86") || arch.contains("amd64") || arch.contains("x86_64");
    }

    private static boolean isArm64(String arch) {
        return arch.contains("aarch64") || arch.contains("arm64");
    }

    // ── Process execution ──────────────────────────────────────────────

    private static String execFirstLine(String... command) throws Exception {
        Process process = new ProcessBuilder(command)
            .redirectErrorStream(true)
            .start();
        try (BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line = reader.readLine();
            process.waitFor();
            return line == null ? "" : line.trim();
        }
    }

    // ── Parsing helpers ────────────────────────────────────────────────

    private static Integer parsePositiveInt(String value) {
        if (value == null) {
            return null;
        }
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static Double parsePositiveDouble(String value) {
        if (value == null) {
            return null;
        }
        String normalized = value.trim().replace(",", "");
        if (normalized.isEmpty()) {
            return null;
        }
        normalized = normalized.replaceAll("[^0-9eE+\\-.]", " ").trim();
        if (normalized.isEmpty()) {
            return null;
        }
        String[] parts = normalized.split("\\s+");
        String candidate = parts.length == 0 ? normalized : parts[0];
        try {
            double parsed = Double.parseDouble(candidate);
            return parsed > 0.0 ? parsed : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    static final class ClockEstimate {
        final double ghz;
        final String source;

        ClockEstimate(double ghz, String source) {
            this.ghz = ghz;
            this.source = source;
        }
    }
}
