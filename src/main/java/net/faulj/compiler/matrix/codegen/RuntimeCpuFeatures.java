package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Runtime CPU gate used before a generated AVX2 function is invoked. */
public final class RuntimeCpuFeatures {
    private static volatile Probe avx2Probe;
    private static volatile Probe fmaProbe;

    private RuntimeCpuFeatures() {
    }

    public static boolean avx2Supported() {
        return cached("avx2", "jlc.compiler.avx2.disabled", true);
    }

    public static boolean fmaSupported() {
        return cached("fma", "jlc.compiler.fma.disabled", false);
    }

    /**
     * CPU feature detection is an initialization-boundary operation. Cache it
     * after the first probe so generated execution never rereads procfs; the
     * property key remains part of the cache key for deterministic tests.
     */
    private static boolean cached(String feature, String disabledProperty,
                                  boolean avx2) {
        String key = feature + "|" + System.getProperty("os.name", "") + "|"
            + System.getProperty("os.arch", "") + "|"
            + System.getProperty(disabledProperty, "false");
        Probe current = avx2 ? avx2Probe : fmaProbe;
        if (current != null && current.key().equals(key)) {
            return current.supported();
        }
        synchronized (RuntimeCpuFeatures.class) {
            current = avx2 ? avx2Probe : fmaProbe;
            if (current != null && current.key().equals(key)) {
                return current.supported();
            }
            boolean supported = detect(feature, disabledProperty);
            Probe next = new Probe(key, supported);
            if (avx2) avx2Probe = next;
            else fmaProbe = next;
            return supported;
        }
    }

    private static boolean detect(String feature, String disabledProperty) {
        if (Boolean.getBoolean(disabledProperty)) {
            return false;
        }
        String architecture = System.getProperty("os.arch", "")
            .toLowerCase(Locale.ROOT);
        if (!(architecture.equals("amd64") || architecture.equals("x86_64")
            || architecture.equals("x86"))) {
            return false;
        }
        if ("linux".equalsIgnoreCase(System.getProperty("os.name", ""))) {
            try {
                String cpuInfo = Files.readString(Path.of("/proc/cpuinfo"))
                    .toLowerCase(Locale.ROOT);
                return cpuInfo.lines()
                    .filter(line -> line.startsWith("flags") || line.startsWith("features"))
                    .anyMatch(line -> line.matches(".*(^|\\s)" + feature + "($|\\s).*"));
            } catch (IOException | SecurityException ignored) {
                return false;
            }
        }
        return false;
    }

    private record Probe(String key, boolean supported) {
    }
}
