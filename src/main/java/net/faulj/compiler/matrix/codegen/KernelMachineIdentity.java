package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Locale;
import java.util.StringJoiner;

/** Stable best-effort machine identity for compiler-kernel calibration. */
public final class KernelMachineIdentity {
    private final String architecture;
    private final String cpuVendor;
    private final String cpuFamily;
    private final String cpuModel;
    private final String cpuModelName;
    private final String isaFeatures;
    private final String os;
    private final String jvm;
    private final String nativeCompiler;
    private final int availableProcessors;
    private final String key;

    public KernelMachineIdentity(String architecture,
                                 String cpuVendor,
                                 String cpuFamily,
                                 String cpuModel,
                                 String cpuModelName,
                                 String isaFeatures,
                                 String os,
                                 String jvm,
                                 String nativeCompiler,
                                 int availableProcessors) {
        this.architecture = text(architecture);
        this.cpuVendor = text(cpuVendor);
        this.cpuFamily = text(cpuFamily);
        this.cpuModel = text(cpuModel);
        this.cpuModelName = text(cpuModelName);
        this.isaFeatures = text(isaFeatures);
        this.os = text(os);
        this.jvm = text(jvm);
        this.nativeCompiler = text(nativeCompiler);
        this.availableProcessors = Math.max(0, availableProcessors);
        this.key = sha256(canonicalText());
    }

    public static KernelMachineIdentity current() {
        CpuInfo info = CpuInfo.read();
        String compiler = firstNonBlank(System.getProperty("jlc.compiler.nativeCompiler"),
            System.getenv("JLC_NATIVE_CXX_COMPILER"), "unknown");
        return new KernelMachineIdentity(
            System.getProperty("os.arch", "unknown"), info.vendor, info.family, info.model,
            info.modelName, info.features, System.getProperty("os.name", "unknown") + "/"
                + System.getProperty("os.version", "unknown"),
            System.getProperty("java.version", "unknown") + "/"
                + System.getProperty("java.vm.name", "unknown") + "/"
                + System.getProperty("java.vm.version", "unknown"), compiler,
            Runtime.getRuntime().availableProcessors());
    }

    public String architecture() { return architecture; }
    public String cpuVendor() { return cpuVendor; }
    public String cpuFamily() { return cpuFamily; }
    public String cpuModel() { return cpuModel; }
    public String cpuModelName() { return cpuModelName; }
    public String isaFeatures() { return isaFeatures; }
    public String os() { return os; }
    public String jvm() { return jvm; }
    public String nativeCompiler() { return nativeCompiler; }
    public int availableProcessors() { return availableProcessors; }
    public String key() { return key; }

    public String canonicalText() {
        return new StringJoiner("\n")
            .add("architecture=" + architecture)
            .add("cpu-vendor=" + cpuVendor)
            .add("cpu-family=" + cpuFamily)
            .add("cpu-model=" + cpuModel)
            .add("cpu-model-name=" + cpuModelName)
            .add("isa-features=" + isaFeatures)
            .add("os=" + os)
            .add("jvm=" + jvm)
            .add("native-compiler=" + nativeCompiler)
            .add("available-processors=" + availableProcessors)
            .toString();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelMachineIdentity identity && key.equals(identity.key);
    }

    @Override
    public int hashCode() { return key.hashCode(); }

    @Override
    public String toString() { return canonicalText(); }

    private static String text(String value) {
        return value == null || value.isBlank() ? "unknown" : value.trim();
    }

    private static String firstNonBlank(String... values) {
        return Arrays.stream(values).filter(value -> value != null && !value.isBlank())
            .findFirst().orElse("unknown");
    }

    private static String sha256(String value) {
        try {
            return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static final class CpuInfo {
        private final String vendor;
        private final String family;
        private final String model;
        private final String modelName;
        private final String features;

        private CpuInfo(String vendor, String family, String model,
                        String modelName, String features) {
            this.vendor = vendor;
            this.family = family;
            this.model = model;
            this.modelName = modelName;
            this.features = features;
        }

        private static CpuInfo read() {
            if (!System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux")) {
                return new CpuInfo("unknown", "unknown", "unknown", "unknown", "unknown");
            }
            try {
                String text = Files.readString(Path.of("/proc/cpuinfo"));
                String vendor = value(text, "vendor_id");
                String family = value(text, "cpu family");
                String model = value(text, "model");
                String modelName = value(text, "model name");
                String flags = firstNonBlank(value(text, "flags"), value(text, "Features"));
                String features = Arrays.stream(flags.toLowerCase(Locale.ROOT).split("\\s+"))
                    .filter(flag -> flag.startsWith("avx") || flag.equals("fma")
                        || flag.equals("sse4_2") || flag.equals("sse4_1"))
                    .distinct().sorted().reduce((left, right) -> left + "," + right)
                    .orElse("none");
                return new CpuInfo(vendor, family, model, modelName, features);
            } catch (IOException | SecurityException ignored) {
                return new CpuInfo("unknown", "unknown", "unknown", "unknown", "unknown");
            }
        }

        private static String value(String text, String key) {
            return text.lines().filter(line -> {
                    int separator = line.indexOf(':');
                    return separator >= 0
                        && line.substring(0, separator).trim().equalsIgnoreCase(key);
                })
                .map(line -> line.substring(line.indexOf(':') + 1).trim())
                .findFirst().orElse("unknown");
        }
    }
}
