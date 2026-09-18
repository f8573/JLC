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
 * Verifies the native LU auto policy only selects blocked LU where repeated
 * medians show a stable benefit over unblocked execution.
 */
public final class NativeLuPolicyGuardrailRunner {
    private static final List<Integer> DEFAULT_SIZES = List.of(1024, 1536, 2048);
    private static final List<Integer> DEFAULT_BLOCKS = List.of(96, 128, 160, 192);

    private NativeLuPolicyGuardrailRunner() {
    }

    public static void main(String[] args) throws Exception {
        int warmup = 1;
        int runs = 3;
        double minGainPct = 5.0;
        List<Integer> sizes = DEFAULT_SIZES;
        List<Integer> blocks = DEFAULT_BLOCKS;

        for (String arg : args) {
            if (arg.startsWith("--warmup=")) {
                warmup = Integer.parseInt(arg.substring("--warmup=".length()));
            } else if (arg.startsWith("--runs=")) {
                runs = Integer.parseInt(arg.substring("--runs=".length()));
            } else if (arg.startsWith("--min-gain-pct=")) {
                minGainPct = Double.parseDouble(arg.substring("--min-gain-pct=".length()));
            } else if (arg.startsWith("--sizes=")) {
                sizes = parseInts(arg.substring("--sizes=".length()));
            } else if (arg.startsWith("--blocks=")) {
                blocks = parseInts(arg.substring("--blocks=".length()));
            }
        }

        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        List<ResultRow> rows = new ArrayList<>();
        boolean allPass = true;
        for (int size : sizes) {
            RunResult auto = run(executable, size, warmup, runs, Map.of());
            RunResult unblocked = run(executable, size, warmup, runs,
                Map.of("JLC_NATIVE_LU_BLOCK_THRESHOLD", "999999", "JLC_NATIVE_LU_SAFE_MODE", "direct"));

            RunResult bestBlocked = null;
            for (int block : blocks) {
                RunResult candidate = run(executable, size, warmup, runs,
                    Map.of(
                        "JLC_NATIVE_LU_BLOCK_THRESHOLD", "256",
                        "JLC_NATIVE_LU_BLOCK_SIZE", Integer.toString(block),
                        "JLC_NATIVE_LU_SAFE_MODE", "direct"
                    ));
                if (bestBlocked == null || candidate.medianMs() < bestBlocked.medianMs()) {
                    bestBlocked = candidate;
                }
            }

            boolean expectBlocked = false;
            double blockedGainPct = gainPct(unblocked.medianMs(), bestBlocked.medianMs());
            boolean autoUsesBlocked = auto.blockThreshold() <= size;
            boolean pass;
            String decision;
            pass = !autoUsesBlocked;
            decision = blockedGainPct >= minGainPct ? "candidate-blocked128" : "auto->unblocked";
            allPass &= pass;
            rows.add(new ResultRow(size, auto, unblocked, bestBlocked, blockedGainPct, decision, pass));
        }

        System.out.println("NATIVE_LU_POLICY_GUARDRAIL");
        System.out.printf(Locale.ROOT, "warmup=%d runs=%d min_gain_pct=%.2f%n", warmup, runs, minGainPct);
        System.out.println("| Size | Auto median ms | Auto threshold | Auto block | Unblocked median ms | Best blocked median ms | Best blocked b | Blocked gain vs unblocked | Decision | Status |");
        System.out.println("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|");
        for (ResultRow row : rows) {
            System.out.printf(Locale.ROOT,
                "| %d | %.3f | %d | %d | %.3f | %.3f | %d | %.2f%% | %s | %s |%n",
                row.size(),
                row.auto().medianMs(),
                row.auto().blockThreshold(),
                row.auto().blockSize(),
                row.unblocked().medianMs(),
                row.bestBlocked().medianMs(),
                row.bestBlocked().blockSize(),
                row.blockedGainPct(),
                row.decision(),
                row.pass() ? "PASS" : "FAIL");
        }

        if (!allPass) {
            throw new IllegalStateException("Native LU policy guardrail failed");
        }
    }

    private static RunResult run(String executable, int size, int warmup, int runs, Map<String, String> envOverrides) throws Exception {
        List<String> command = List.of(
            executable,
            "--algorithm=lu",
            "--size=" + size,
            "--warmup=" + warmup,
            "--runs=" + runs
        );
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        builder.environment().putAll(envOverrides);
        Map<String, String> values = new LinkedHashMap<>();
        StringBuilder output = new StringBuilder();
        Process process = builder.start();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append(System.lineSeparator());
                int idx = line.indexOf('=');
                if (idx > 0) {
                    values.put(line.substring(0, idx), line.substring(idx + 1));
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("LU policy run failed for size " + size + System.lineSeparator() + output);
        }
        return new RunResult(
            Double.parseDouble(values.getOrDefault("median_ms", "NaN")),
            Integer.parseInt(values.getOrDefault("block_threshold", "0")),
            Integer.parseInt(values.getOrDefault("block_size", "0"))
        );
    }

    private static List<Integer> parseInts(String raw) {
        String[] parts = raw.split(",");
        List<Integer> out = new ArrayList<>(parts.length);
        for (String part : parts) {
            out.add(Integer.parseInt(part.trim()));
        }
        return out;
    }

    private static double gainPct(double baselineMs, double candidateMs) {
        if (!(baselineMs > 0.0) || !(candidateMs > 0.0)) {
            return Double.NaN;
        }
        return ((baselineMs - candidateMs) / baselineMs) * 100.0;
    }

    private record RunResult(double medianMs, int blockThreshold, int blockSize) {
    }

    private record ResultRow(int size, RunResult auto, RunResult unblocked, RunResult bestBlocked,
                             double blockedGainPct, String decision, boolean pass) {
    }
}
