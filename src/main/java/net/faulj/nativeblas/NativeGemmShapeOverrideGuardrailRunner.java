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
 * Verifies that encoded shape overrides beat the legacy auto heuristic by a
 * minimum median improvement threshold before we rely on them.
 */
public final class NativeGemmShapeOverrideGuardrailRunner {
    private static final List<OverrideCase> CASES = List.of(
        new OverrideCase("square-512", 512, 512, 512, 0, 0, 0, false),
        new OverrideCase("square-1024", 1024, 1024, 1024, 0, 0, 0, false),
        new OverrideCase("square-2048", 2048, 2048, 2048, 0, 0, 0, false),
        new OverrideCase("tall-skinny", 2048, 256, 128, 0, 0, 0, false),
        new OverrideCase("wide", 128, 256, 2048, 0, 0, 0, false),
        new OverrideCase("rectangular", 512, 2048, 256, 0, 0, 0, false)
    );

    private NativeGemmShapeOverrideGuardrailRunner() {
    }

    public static void main(String[] args) throws Exception {
        int warmup = 2;
        int runs = 5;
        int threads = 1;
        double minGainPct = 4.0;

        for (String arg : args) {
            if (arg.startsWith("--warmup=")) {
                warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            } else if (arg.startsWith("--runs=")) {
                runs = Integer.parseInt(arg.substring("--runs=".length()));
            } else if (arg.startsWith("--threads=")) {
                threads = Integer.parseInt(arg.substring("--threads=".length()));
            } else if (arg.startsWith("--min-gain-pct=")) {
                minGainPct = Double.parseDouble(arg.substring("--min-gain-pct=".length()));
            }
        }

        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        List<GuardResult> results = new ArrayList<>();
        for (OverrideCase testCase : CASES) {
            RunResult baseline = run(executable, testCase, warmup, runs, threads, true);
            RunResult override = testCase.expectOverride()
                ? run(executable, testCase, warmup, runs, threads, false)
                : baseline;
            double gainPct = baseline.medianGflops() == 0.0
                ? 0.0
                : ((override.medianGflops() - baseline.medianGflops()) / baseline.medianGflops()) * 100.0;
            boolean pass = !testCase.expectOverride() || gainPct >= minGainPct;
            results.add(new GuardResult(testCase, baseline, override, gainPct, pass));
        }

        System.out.println("NATIVE_GEMM_SHAPE_OVERRIDE_GUARDRAIL");
        System.out.printf(Locale.ROOT, "warmup=%d runs=%d threads=%d min_gain_pct=%.2f%n", warmup, runs, threads, minGainPct);
        System.out.println("| Shape | Baseline Median GFLOPs | Override Median GFLOPs | Gain | Expect Override | Status |");
        System.out.println("|---|---:|---:|---:|---|---|");
        boolean allPass = true;
        for (GuardResult result : results) {
            allPass &= result.pass();
            System.out.printf(Locale.ROOT, "| %s | %.3f | %.3f | %.2f%% | %s | %s |%n",
                result.testCase().name(),
                result.baseline().medianGflops(),
                result.override().medianGflops(),
                result.gainPct(),
                result.testCase().expectOverride() ? "yes" : "no",
                result.pass() ? "PASS" : "FAIL");
        }
        if (!allPass) {
            throw new IllegalStateException("One or more shape overrides failed the median gain guardrail");
        }
    }

    private static RunResult run(String executable, OverrideCase testCase, int warmup, int runs, int threads,
                                 boolean disableShapeOverrides) throws Exception {
        List<String> command = List.of(
            executable,
            "--algorithm=gemm",
            "--rows=" + testCase.rows(),
            "--inner=" + testCase.inner(),
            "--cols=" + testCase.cols(),
            "--warmup=" + warmup,
            "--runs=" + runs,
            "--threads=" + threads,
            "--diag"
        );
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> env = builder.environment();
        env.remove("JLC_NATIVE_MC");
        env.remove("JLC_NATIVE_NC");
        env.remove("JLC_NATIVE_KC");
        if (disableShapeOverrides) {
            env.put("JLC_NATIVE_DISABLE_SHAPE_OVERRIDES", "1");
        } else {
            env.remove("JLC_NATIVE_DISABLE_SHAPE_OVERRIDES");
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
            throw new IllegalStateException("Shape override guardrail run failed for " + testCase.name()
                + System.lineSeparator() + output);
        }
        return new RunResult(
            Double.parseDouble(diag.getOrDefault("median_ms", "NaN")),
            Double.parseDouble(diag.getOrDefault("median_gflops", "NaN")),
            Long.parseLong(diag.getOrDefault("mc", "0")),
            Long.parseLong(diag.getOrDefault("nc", "0")),
            Long.parseLong(diag.getOrDefault("kc", "0"))
        );
    }

    private record OverrideCase(String name, int rows, int inner, int cols, int expectedMc, int expectedNc,
                                int expectedKc, boolean expectOverride) {
    }

    private record RunResult(double medianMs, double medianGflops, long mc, long nc, long kc) {
    }

    private record GuardResult(OverrideCase testCase, RunResult baseline, RunResult override, double gainPct,
                               boolean pass) {
    }
}
