package net.faulj.nativeblas;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Focused probe for array-backed JNI GEMM access overhead versus direct-buffer JNI at one shape.
 */
public final class NativeGemmJniArrayAccessProbeRunner {
    private static final int LOOP_DEFAULT = 0;
    private static final int LOOP_JC_PC_IC = 1;

    private NativeGemmJniArrayAccessProbeRunner() {
    }

    public static void main(String[] args) {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int warmup = 5;
        int runs = 15;
        int threads = 1;
        long seed = 17_512L;

        for (String arg : args) {
            if (arg.startsWith("--rows=")) rows = Integer.parseInt(arg.substring("--rows=".length()));
            else if (arg.startsWith("--inner=")) inner = Integer.parseInt(arg.substring("--inner=".length()));
            else if (arg.startsWith("--cols=")) cols = Integer.parseInt(arg.substring("--cols=".length()));
            else if (arg.startsWith("--warmup=")) warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            else if (arg.startsWith("--runs=")) runs = Integer.parseInt(arg.substring("--runs=".length()));
            else if (arg.startsWith("--threads=")) threads = Integer.parseInt(arg.substring("--threads=".length()));
            else if (arg.startsWith("--seed=")) seed = Long.parseLong(arg.substring("--seed=".length()));
        }

        double[] aSeed = randomData(rows, inner, seed);
        double[] bSeed = randomData(inner, cols, seed + 1);

        List<Variant> variants = List.of(
            new Variant("current", 0, 0, 0, false, LOOP_DEFAULT),
            new Variant("candidate_nc48", 128, 512, 48, true, LOOP_JC_PC_IC),
            new Variant("candidate_nc32", 128, 512, 32, true, LOOP_JC_PC_IC)
        );

        System.out.println("NATIVE_GEMM_JNI_ARRAY_ACCESS_PROBE");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d warmup=%d runs=%d threads=%d%n", rows, inner, inner, cols, warmup, runs, threads);
        System.out.println();
        System.out.println("| Variant | Path | Median ms | P25 ms | P75 ms | Acquire avg ms | Native avg ms | Release avg ms | Extra avg ms |");
        System.out.println("|---|---|---:|---:|---:|---:|---:|---:|---:|");

        for (Variant variant : variants) {
            ProbeResult arrayResult = measureArrayVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant);
            System.out.printf(Locale.ROOT, "| %s | array-direct | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
                variant.label(),
                arrayResult.timing().medianMs(),
                arrayResult.timing().p25Ms(),
                arrayResult.timing().p75Ms(),
                nanosToMs(arrayResult.profile().acquireNanos()) / runs,
                nanosToMs(arrayResult.profile().nativeNanos()) / runs,
                nanosToMs(arrayResult.profile().releaseNanos()) / runs,
                arrayResult.timing().meanMs() - (nanosToMs(arrayResult.profile().nativeNanos()) / runs)
            );
            if (variant == variants.getFirst()) {
                printAlignment("array-pinned", arrayResult.profile(), true);
            }
            ProbeResult scratchResult = measureArrayScratchVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variant);
            System.out.printf(Locale.ROOT, "| %s | array-copy-aligned-scratch | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
                variant.label(),
                scratchResult.timing().medianMs(),
                scratchResult.timing().p25Ms(),
                scratchResult.timing().p75Ms(),
                0.0,
                0.0,
                0.0,
                scratchResult.timing().meanMs()
            );
        }

        ProbeResult directCurrent = measureDirectVariant(aSeed, bSeed, rows, inner, cols, warmup, runs, threads, variants.getFirst());
        System.out.printf(Locale.ROOT, "| %s | direct-native-buffer | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |%n",
            variants.getFirst().label(),
            directCurrent.timing().medianMs(),
            directCurrent.timing().p25Ms(),
            directCurrent.timing().p75Ms(),
            0.0,
            0.0,
            0.0,
            0.0
        );
        printAlignment("direct-buffer", directCurrent.profile(), false);
    }

    private static ProbeResult measureArrayVariant(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                                   int warmup, int runs, int threads, Variant variant) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        double[] a = aSeed.clone();
        double[] b = bSeed.clone();
        double[] c = new double[rows * cols];

        applyVariant(variant);
        try {
            for (int i = 0; i < warmup; i++) {
                java.util.Arrays.fill(c, 0.0);
                NativeBindings.nativeGemm(a, rows, inner, b, inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
            }

            NativeProfiling.setJniGemmArrayEnabled(true);
            NativeProfiling.resetJniGemmArray();
            List<Double> samples = new ArrayList<>();
            for (int i = 0; i < runs; i++) {
                java.util.Arrays.fill(c, 0.0);
                long start = System.nanoTime();
                NativeBindings.nativeGemm(a, rows, inner, b, inner, cols, c, rows, cols, 1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN);
                samples.add((System.nanoTime() - start) / 1e6);
            }
            NativeJniGemmArrayProfile profile = NativeProfiling.jniGemmArraySnapshot().orElse(NativeJniGemmArrayProfile.EMPTY);
            NativeProfiling.setJniGemmArrayEnabled(false);
            return new ProbeResult(summarize(samples), profile);
        } finally {
            clearVariant();
            NativeProfiling.setJniGemmArrayEnabled(false);
        }
    }

    private static ProbeResult measureDirectVariant(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                                    int warmup, int runs, int threads, Variant variant) {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        try (NativeMatrix a = NativeMatrix.allocate(rows, inner);
             NativeMatrix b = NativeMatrix.allocate(inner, cols);
             NativeMatrix c = NativeMatrix.allocate(rows, cols)) {
            fillNativeMatrix(a, aSeed);
            fillNativeMatrix(b, bSeed);

            applyVariant(variant);
            try {
                NativeProfiling.setJniGemmArrayEnabled(true);
                NativeProfiling.resetJniGemmArray();
                List<Double> samples = new ArrayList<>();
                for (int i = 0; i < warmup + runs; i++) {
                    zeroNativeMatrix(c);
                    long start = System.nanoTime();
                    NativeBindings.nativeGemmDirect(
                        a.buffer(), 0L, a.ld(), rows, inner, 0,
                        b.buffer(), 0L, b.ld(), inner, cols, 0,
                        c.buffer(), 0L, c.ld(), rows, cols, 0,
                        1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN
                    );
                    if (i >= warmup) {
                        samples.add((System.nanoTime() - start) / 1e6);
                    }
                }
                NativeJniGemmArrayProfile profile = NativeProfiling.jniGemmArraySnapshot().orElse(NativeJniGemmArrayProfile.EMPTY);
                NativeProfiling.setJniGemmArrayEnabled(false);
                return new ProbeResult(summarize(samples), profile);
            } finally {
                clearVariant();
                NativeProfiling.setJniGemmArrayEnabled(false);
            }
        }
    }

    private static ProbeResult measureArrayScratchVariant(double[] aSeed, double[] bSeed, int rows, int inner, int cols,
                                                          int warmup, int runs, int threads, Variant variant) {
        try (NativeMatrix a = NativeMatrix.allocate(rows, inner);
             NativeMatrix b = NativeMatrix.allocate(inner, cols);
             NativeMatrix c = NativeMatrix.allocate(rows, cols)) {
            double[] cScratch = new double[rows * cols];
            applyVariant(variant);
            try {
                List<Double> samples = new ArrayList<>();
                for (int i = 0; i < warmup + runs; i++) {
                    fillNativeMatrix(a, aSeed);
                    fillNativeMatrix(b, bSeed);
                    zeroNativeMatrix(c);
                    long start = System.nanoTime();
                    NativeBindings.nativeGemmDirect(
                        a.buffer(), 0L, a.ld(), rows, inner, 0,
                        b.buffer(), 0L, b.ld(), inner, cols, 0,
                        c.buffer(), 0L, c.ld(), rows, cols, 0,
                        1.0, 0.0, threads, NativeFlags.FORCE_BUILTIN
                    );
                    readNativeMatrix(c, cScratch);
                    if (i >= warmup) {
                        samples.add((System.nanoTime() - start) / 1e6);
                    }
                }
                return new ProbeResult(summarize(samples), NativeJniGemmArrayProfile.EMPTY);
            } finally {
                clearVariant();
            }
        }
    }

    private static void applyVariant(Variant variant) {
        if (variant.disableSquareTuning()) {
            NativeBindings.nativeGemmSetRuntimeOverrides(
                variant.mc(), variant.kc(), variant.nc(),
                0, 0,
                true,
                variant.loopOrder()
            );
        } else {
            NativeBindings.nativeGemmClearRuntimeOverrides();
        }
    }

    private static void clearVariant() {
        NativeBindings.nativeGemmClearRuntimeOverrides();
    }

    private static void fillNativeMatrix(NativeMatrix matrix, double[] values) {
        DoubleBuffer buffer = matrix.asDoubleBuffer();
        buffer.position(0);
        buffer.put(values);
        buffer.position(0);
    }

    private static void zeroNativeMatrix(NativeMatrix matrix) {
        DoubleBuffer buffer = matrix.asDoubleBuffer();
        buffer.position(0);
        for (int i = 0; i < matrix.rows() * matrix.cols(); i++) {
            buffer.put(0.0);
        }
        buffer.position(0);
    }

    private static void readNativeMatrix(NativeMatrix matrix, double[] output) {
        DoubleBuffer buffer = matrix.asDoubleBuffer();
        buffer.position(0);
        buffer.get(output);
        buffer.position(0);
    }

    private static void printAlignment(String label, NativeJniGemmArrayProfile profile, boolean arrayPath) {
        if (arrayPath) {
            System.out.printf(Locale.ROOT,
                "alignment[%s]: A(mod16/mod32/mod64)=%d/%d/%d B=%d/%d/%d C=%d/%d/%d%n",
                label,
                profile.arrayAMod16(), profile.arrayAMod32(), profile.arrayAMod64(),
                profile.arrayBMod16(), profile.arrayBMod32(), profile.arrayBMod64(),
                profile.arrayCMod16(), profile.arrayCMod32(), profile.arrayCMod64()
            );
        } else {
            System.out.printf(Locale.ROOT,
                "alignment[%s]: A(mod16/mod32/mod64)=%d/%d/%d B=%d/%d/%d C=%d/%d/%d%n",
                label,
                profile.directAMod16(), profile.directAMod32(), profile.directAMod64(),
                profile.directBMod16(), profile.directBMod32(), profile.directBMod64(),
                profile.directCMod16(), profile.directCMod32(), profile.directCMod64()
            );
        }
    }

    private static Timing summarize(List<Double> samples) {
        List<Double> sorted = new ArrayList<>(samples);
        Collections.sort(sorted);
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double total = 0.0;
        for (double sample : sorted) {
            min = Math.min(min, sample);
            max = Math.max(max, sample);
            total += sample;
        }
        int middle = sorted.size() / 2;
        double median = (sorted.size() % 2 == 0)
            ? 0.5 * (sorted.get(middle - 1) + sorted.get(middle))
            : sorted.get(middle);
        double p25 = percentile(sorted, 0.25);
        double p75 = percentile(sorted, 0.75);
        return new Timing(min, max, total / sorted.size(), median, p25, p75);
    }

    private static double percentile(List<Double> sorted, double p) {
        if (sorted.isEmpty()) {
            return Double.NaN;
        }
        double index = p * (sorted.size() - 1);
        int lo = (int) Math.floor(index);
        int hi = (int) Math.ceil(index);
        if (lo == hi) {
            return sorted.get(lo);
        }
        double weight = index - lo;
        return sorted.get(lo) * (1.0 - weight) + sorted.get(hi) * weight;
    }

    private static double[] randomData(int rows, int cols, long seed) {
        Random random = new Random(seed);
        double[] data = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return data;
    }

    private static double nanosToMs(long nanos) {
        return nanos / 1e6;
    }

    private record Variant(String label, int mc, int kc, int nc, boolean disableSquareTuning, int loopOrder) {
    }

    private record Timing(double minMs, double maxMs, double meanMs, double medianMs, double p25Ms, double p75Ms) {
    }

    private record ProbeResult(Timing timing, NativeJniGemmArrayProfile profile) {
    }
}
