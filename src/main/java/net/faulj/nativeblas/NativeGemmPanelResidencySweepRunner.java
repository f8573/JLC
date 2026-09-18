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
 * Sweeps KC/MC/NC and loop-order variants to explain the panel-walk-to-full-GEMM gap.
 */
public final class NativeGemmPanelResidencySweepRunner {
    private static final int[] DEFAULT_KC = {64, 128, 256, 512};
    private static final int[] DEFAULT_MC = {64, 128, 256};
    private static final int[] DEFAULT_NC = {64, 128, 256};
    private static final String[] DEFAULT_VARIANTS = {"jc_pc_ic", "ic_pc_jc"};

    private NativeGemmPanelResidencySweepRunner() {
    }

    public static void main(String[] args) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int warmup = 1;
        int runs = 2;
        int threads = 1;
        int[] kcValues = DEFAULT_KC;
        int[] mcValues = DEFAULT_MC;
        int[] ncValues = DEFAULT_NC;
        String[] variants = DEFAULT_VARIANTS;

        for (String arg : args) {
            if (arg.startsWith("--rows=")) rows = Integer.parseInt(arg.substring("--rows=".length()));
            else if (arg.startsWith("--inner=")) inner = Integer.parseInt(arg.substring("--inner=".length()));
            else if (arg.startsWith("--cols=")) cols = Integer.parseInt(arg.substring("--cols=".length()));
            else if (arg.startsWith("--warmup=")) warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            else if (arg.startsWith("--runs=")) runs = Integer.parseInt(arg.substring("--runs=".length()));
            else if (arg.startsWith("--threads=")) threads = Integer.parseInt(arg.substring("--threads=".length()));
            else if (arg.startsWith("--kc=")) kcValues = parseCsvInts(arg.substring("--kc=".length()));
            else if (arg.startsWith("--mc=")) mcValues = parseCsvInts(arg.substring("--mc=".length()));
            else if (arg.startsWith("--nc=")) ncValues = parseCsvInts(arg.substring("--nc=".length()));
            else if (arg.startsWith("--variants=")) variants = parseCsvStrings(arg.substring("--variants=".length()));
        }

        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        List<Result> results = new ArrayList<>();
        for (String variant : variants) {
            for (int kc : kcValues) {
                for (int mc : mcValues) {
                    for (int nc : ncValues) {
                        results.add(run(executable, rows, inner, cols, warmup, runs, threads, variant, kc, mc, nc));
                    }
                }
            }
        }

        System.out.println("NATIVE_GEMM_PANEL_RESIDENCY_SWEEP");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d warmup=%d runs=%d threads=%d%n", rows, inner, inner, cols, warmup, runs, threads);
        System.out.println();
        System.out.println("| Variant | KC | MC | NC | Min ms | Median ms | Max ms | Mean ms | Median GFLOPs | PackA % | PackB % | Alloc % | Kernel-only % | C/beta % | Edge % | Other % |");
        System.out.println("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for (Result result : results) {
            System.out.printf(Locale.ROOT,
                "| %s | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |%n",
                result.variant,
                result.kc,
                result.mc,
                result.nc,
                result.minMs,
                result.medianMs,
                result.maxMs,
                result.meanMs,
                result.medianGflops,
                result.packAPct,
                result.packBPct,
                result.allocPct,
                result.kernelOnlyPct,
                result.scalePct,
                result.edgePct,
                result.otherPct
            );
        }

        Result best = null;
        for (Result result : results) {
            if (best == null || result.medianGflops > best.medianGflops) {
                best = result;
            }
        }
        if (best != null) {
            System.out.println();
            System.out.printf(Locale.ROOT,
                "best=%s KC=%d MC=%d NC=%d median=%.3fms median_gflops=%.3f other=%.1f%%%n",
                best.variant, best.kc, best.mc, best.nc, best.medianMs, best.medianGflops, best.otherPct);
        }
    }

    private static Result run(String executable, int rows, int inner, int cols, int warmup, int runs, int threads,
                              String variant, int kc, int mc, int nc) throws Exception {
        List<String> command = List.of(
            executable,
            "--algorithm=gemm",
            "--rows=" + rows,
            "--inner=" + inner,
            "--cols=" + cols,
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--diag"
        );
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = builder.environment();
        env.put("JLC_NATIVE_LOOP_ORDER", variant);
        env.put("JLC_NATIVE_KC", Integer.toString(kc));
        env.put("JLC_NATIVE_MC", Integer.toString(mc));
        env.put("JLC_NATIVE_NC", Integer.toString(nc));

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
            throw new IllegalStateException("Panel residency sweep failed for " + variant + " kc=" + kc + " mc=" + mc + " nc=" + nc
                + System.lineSeparator() + output);
        }

        double minMs = Double.parseDouble(diag.getOrDefault("best_ms", "NaN"));
        double medianMs = Double.parseDouble(diag.getOrDefault("median_ms", "NaN"));
        double maxMs = Double.parseDouble(diag.getOrDefault("max_ms", "NaN"));
        double meanMs = Double.parseDouble(diag.getOrDefault("mean_ms", "NaN"));
        double medianGflops = Double.parseDouble(diag.getOrDefault("median_gflops", "NaN"));
        long wallNs = longValue(diag, "wall_ns");
        long packA = longValue(diag, "pack_a_ns");
        long packB = longValue(diag, "pack_b_ns");
        long alloc = longValue(diag, "alloc_ns");
        long kernel = longValue(diag, "kernel_ns");
        long scale = longValue(diag, "scale_c_ns");
        long edge = longValue(diag, "edge_ns");
        long kernelOnly = Math.max(0L, kernel - edge);
        long accounted = packA + packB + alloc + kernelOnly + scale + edge;
        long other = Math.max(0L, wallNs - accounted);

        return new Result(
            variant, kc, mc, nc, minMs, medianMs, maxMs, meanMs, medianGflops,
            pct(packA, wallNs),
            pct(packB, wallNs),
            pct(alloc, wallNs),
            pct(kernelOnly, wallNs),
            pct(scale, wallNs),
            pct(edge, wallNs),
            pct(other, wallNs)
        );
    }

    private static long longValue(Map<String, String> diag, String key) {
        return Long.parseLong(diag.getOrDefault(key, "0"));
    }

    private static double pct(long numer, long denom) {
        return denom == 0L ? 0.0 : numer * 100.0 / denom;
    }

    private static int[] parseCsvInts(String value) {
        String[] parts = value.split(",");
        int[] parsed = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            parsed[i] = Integer.parseInt(parts[i].trim());
        }
        return parsed;
    }

    private static String[] parseCsvStrings(String value) {
        String[] parts = value.split(",");
        for (int i = 0; i < parts.length; i++) {
            parts[i] = parts[i].trim();
        }
        return parts;
    }

    private record Result(String variant, int kc, int mc, int nc,
                          double minMs, double medianMs, double maxMs, double meanMs, double medianGflops,
                          double packAPct, double packBPct, double allocPct, double kernelOnlyPct,
                          double scalePct, double edgePct, double otherPct) {
    }
}
