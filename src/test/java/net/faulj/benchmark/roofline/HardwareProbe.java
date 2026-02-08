package net.faulj.benchmark.roofline;

import jdk.incubator.vector.DoubleVector;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

final class HardwareProbe {
    private static final String CLOCK_PROPERTY = "jlc.roofline.cpu_ghz";
    private static final String CLOCK_ENV = "JLC_ROOFLINE_CPU_GHZ";
    private static final String FMA_PROPERTY = "jlc.roofline.fma";
    private static final String FMA_ENV = "JLC_ROOFLINE_FMA";
    private static final String ISSUE_WIDTH_PROPERTY = "jlc.roofline.vector_issue_width";
    private static final String ISSUE_WIDTH_ENV = "JLC_ROOFLINE_VECTOR_ISSUE_WIDTH";

    private HardwareProbe() {
    }

    static HardwareInfo probe() {
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        int simdLanes = detectSimdLanes();
        boolean fma = detectFma();
        int issueWidth = detectIssueWidth();
        ClockEstimate clockEstimate = detectClockGhz();

        double flopsPerCyclePerCore = simdLanes * (fma ? 2.0 : 1.0) * issueWidth;
        double peakFlops = cores * clockEstimate.ghz * 1e9 * flopsPerCyclePerCore;

        return new HardwareInfo(
            cores,
            simdLanes,
            fma,
            clockEstimate.ghz,
            issueWidth,
            peakFlops,
            clockEstimate.source
        );
    }

    private static int detectSimdLanes() {
        try {
            return Math.max(1, DoubleVector.SPECIES_PREFERRED.length());
        } catch (Throwable ignored) {
            String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
            if (arch.contains("x86") || arch.contains("amd64") || arch.contains("aarch64") || arch.contains("arm64")) {
                return 2;
            }
            return 1;
        }
    }

    private static boolean detectFma() {
        Boolean explicit = BenchmarkMode.parseFlag(System.getProperty(FMA_PROPERTY));
        if (explicit != null) {
            return explicit;
        }
        explicit = BenchmarkMode.parseFlag(System.getenv(FMA_ENV));
        if (explicit != null) {
            return explicit;
        }
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        return arch.contains("x86") || arch.contains("amd64") || arch.contains("aarch64") || arch.contains("arm64");
    }

    private static int detectIssueWidth() {
        Integer fromProperty = parsePositiveInt(System.getProperty(ISSUE_WIDTH_PROPERTY));
        if (fromProperty != null) {
            return fromProperty;
        }
        Integer fromEnv = parsePositiveInt(System.getenv(ISSUE_WIDTH_ENV));
        if (fromEnv != null) {
            return fromEnv;
        }
        return 1;
    }

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

    private static final class ClockEstimate {
        final double ghz;
        final String source;

        ClockEstimate(double ghz, String source) {
            this.ghz = ghz;
            this.source = source;
        }
    }
}
