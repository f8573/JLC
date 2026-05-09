package net.faulj.nativeblas;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Sweeps native GEMM block sizes across representative shapes.
 */
public final class NativeGemmBlockSizeSweepRunner {
    private static final List<Shape> DEFAULT_SHAPES = List.of(
        new Shape("square-512", 512, 512, 512),
        new Shape("square-1024", 1024, 1024, 1024),
        new Shape("square-2048", 2048, 2048, 2048),
        new Shape("tall-skinny", 2048, 256, 128),
        new Shape("wide", 128, 256, 2048),
        new Shape("rectangular", 512, 2048, 256)
    );

    private static final int[] DEFAULT_MC = {512, 1024, 2048};
    private static final int[] DEFAULT_NC = {64, 128, 256};
    private static final int[] DEFAULT_KC = {64, 128, 256};

    private NativeGemmBlockSizeSweepRunner() {
    }

    public static void main(String[] args) throws Exception {
        int warmup = 2;
        int runs = 3;
        int threads = 1;
        boolean includeAuto = true;
        String shapeFilter = "";
        int[] mcValues = DEFAULT_MC;
        int[] ncValues = DEFAULT_NC;
        int[] kcValues = DEFAULT_KC;

        for (String arg : args) {
            if (arg.startsWith("--warmup=")) {
                warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            } else if (arg.startsWith("--runs=")) {
                runs = Integer.parseInt(arg.substring("--runs=".length()));
            } else if (arg.startsWith("--threads=")) {
                threads = Integer.parseInt(arg.substring("--threads=".length()));
            } else if (arg.startsWith("--shape=")) {
                shapeFilter = arg.substring("--shape=".length()).trim().toLowerCase(Locale.ROOT);
            } else if (arg.startsWith("--mc=")) {
                mcValues = parseIntList(arg.substring("--mc=".length()));
            } else if (arg.startsWith("--nc=")) {
                ncValues = parseIntList(arg.substring("--nc=".length()));
            } else if (arg.startsWith("--kc=")) {
                kcValues = parseIntList(arg.substring("--kc=".length()));
            } else if (arg.equals("--no-auto")) {
                includeAuto = false;
            }
        }

        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        final String requestedShapeFilter = shapeFilter;
        List<Shape> shapes = DEFAULT_SHAPES.stream()
            .filter(shape -> requestedShapeFilter.isBlank() || shape.name().contains(requestedShapeFilter))
            .toList();
        if (shapes.isEmpty()) {
            throw new IllegalArgumentException("No shapes matched filter: " + shapeFilter);
        }

        List<Result> allResults = new ArrayList<>();
        for (Shape shape : shapes) {
            if (includeAuto) {
                allResults.add(runShape(executable, shape, warmup, runs, threads, 0, 0, 0));
            }
            for (int mc : mcValues) {
                for (int nc : ncValues) {
                    for (int kc : kcValues) {
                        allResults.add(runShape(executable, shape, warmup, runs, threads, mc, nc, kc));
                    }
                }
            }
        }

        System.out.println("NATIVE_GEMM_BLOCK_SIZE_SWEEP");
        System.out.printf(Locale.ROOT, "warmup=%d runs=%d threads=%d%n", warmup, runs, threads);
        System.out.println();
        System.out.println("| Shape | MC | NC | KC | Best ms | Median ms | GFLOPs(best) | GFLOPs(median) | Kernel | Fallback | AVX2 |");
        System.out.println("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|");
        for (Result result : allResults) {
            System.out.printf(Locale.ROOT, "| %s | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |%n",
                result.shape().name(),
                blockLabel(result.mc()),
                blockLabel(result.nc()),
                blockLabel(result.kc()),
                result.bestMs(),
                result.medianMs(),
                result.bestGflops(),
                result.medianGflops(),
                result.selectedKernel(),
                result.scalarFallback() ? "true" : "false",
                result.avx2Enabled() ? "true" : "false");
        }

        System.out.println();
        System.out.println("| Shape | Winner MC | Winner NC | Winner KC | Winner Median GFLOPs | Auto Median GFLOPs | Gain |");
        System.out.println("|---|---:|---:|---:|---:|---:|---:|");
        for (Shape shape : shapes) {
            Result best = null;
            Result auto = null;
            for (Result result : allResults) {
                if (!result.shape().equals(shape)) {
                    continue;
                }
                if (result.isAuto()) {
                    auto = result;
                }
                if (best == null || result.medianGflops() > best.medianGflops()) {
                    best = result;
                }
            }
            if (best == null) {
                continue;
            }
            double autoGflops = auto == null ? Double.NaN : auto.medianGflops();
            double gainPct = auto == null || auto.medianGflops() == 0.0
                ? Double.NaN
                : ((best.medianGflops() - auto.medianGflops()) / auto.medianGflops()) * 100.0;
            System.out.printf(Locale.ROOT, "| %s | %s | %s | %s | %.3f | %s | %s |%n",
                shape.name(),
                blockLabel(best.mc()),
                blockLabel(best.nc()),
                blockLabel(best.kc()),
                best.medianGflops(),
                auto == null ? "n/a" : String.format(Locale.ROOT, "%.3f", autoGflops),
                Double.isNaN(gainPct) ? "n/a" : String.format(Locale.ROOT, "%.2f%%", gainPct));
        }
    }

    private static Result runShape(String executable, Shape shape, int warmup, int runs, int threads,
                                   int mc, int nc, int kc) throws Exception {
        List<String> command = List.of(
            executable,
            "--algorithm=gemm",
            "--rows=" + shape.rows(),
            "--inner=" + shape.inner(),
            "--cols=" + shape.cols(),
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--diag"
        );
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = builder.environment();
        if (mc > 0) {
            env.put("JLC_NATIVE_MC", Integer.toString(mc));
        } else {
            env.remove("JLC_NATIVE_MC");
        }
        if (nc > 0) {
            env.put("JLC_NATIVE_NC", Integer.toString(nc));
        } else {
            env.remove("JLC_NATIVE_NC");
        }
        if (kc > 0) {
            env.put("JLC_NATIVE_KC", Integer.toString(kc));
        } else {
            env.remove("JLC_NATIVE_KC");
        }

        Process process = builder.start();
        Map<String, String> diag = new LinkedHashMap<>();
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append(System.lineSeparator());
                int idx = line.indexOf('=');
                if (idx > 0) {
                    diag.put(line.substring(0, idx), line.substring(idx + 1));
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("Native GEMM sweep command failed for " + shape.name()
                + " mc=" + mc + " nc=" + nc + " kc=" + kc + System.lineSeparator() + output);
        }

        return new Result(
            shape,
            mc,
            nc,
            kc,
            Double.parseDouble(diag.getOrDefault("best_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("median_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("best_gflops", "NaN")),
            Double.parseDouble(diag.getOrDefault("median_gflops", "NaN")),
            diag.getOrDefault("selected_kernel", "unknown"),
            Boolean.parseBoolean(diag.getOrDefault("scalar_fallback", "false")),
            Boolean.parseBoolean(diag.getOrDefault("avx2_enabled", "false"))
        );
    }

    private static int[] parseIntList(String raw) {
        String[] parts = raw.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    private static String blockLabel(int value) {
        return value <= 0 ? "auto" : Integer.toString(value);
    }

    private record Shape(String name, int rows, int inner, int cols) {
    }

    private record Result(Shape shape, int mc, int nc, int kc, double bestMs, double medianMs,
                          double bestGflops, double medianGflops,
                          String selectedKernel, boolean scalarFallback, boolean avx2Enabled) {
        boolean isAuto() {
            return mc == 0 && nc == 0 && kc == 0;
        }
    }
}
