package net.faulj.nativeblas;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Creates a reproducible native GEMM forensic artifact that separates microkernel, packing, and full blocked GEMM.
 */
public final class NativeGemmDeepDiveRunner {
    private NativeGemmDeepDiveRunner() {
    }

    public static void main(String[] args) throws Exception {
        int[] sizes = {128, 512, 1024, 2048};
        int threads = 1;
        int microkernelReps = 250_000;
        Path output = Path.of("native_gemm_deep_dive_" + LocalDate.now().toString().replace('-', '_') + ".md");

        for (String arg : args) {
            if (arg.startsWith("--sizes=")) {
                sizes = parseIntList(arg.substring("--sizes=".length()));
            } else if (arg.startsWith("--threads=")) {
                threads = Integer.parseInt(arg.substring("--threads=".length()));
            } else if (arg.startsWith("--microkernelReps=")) {
                microkernelReps = Integer.parseInt(arg.substring("--microkernelReps=".length()));
            } else if (arg.startsWith("--output=")) {
                output = Path.of(arg.substring("--output=".length()));
            }
        }

        String executable = System.getProperty("jlc.native.gemm.deep.dive.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.gemm.deep.dive.path");
        }

        List<String> variants = List.of("jc_pc_ic", "ic_pc_jc");
        List<Result> results = new ArrayList<>();
        for (int size : sizes) {
            for (String variant : variants) {
                results.add(runCase(executable, size, threads, microkernelReps, variant));
            }
        }

        String markdown = renderMarkdown(results, threads, microkernelReps);
        Files.writeString(output, markdown, StandardCharsets.UTF_8);
        System.out.println("Wrote native GEMM deep dive to " + output.toAbsolutePath());
    }

    private static Result runCase(String executable, int size, int threads, int microkernelReps, String variant) throws Exception {
        ProcessBuilder builder = new ProcessBuilder(
            executable,
            "--rows=" + size,
            "--inner=" + size,
            "--cols=" + size,
            "--threads=" + threads,
            "--microkernel-reps=" + microkernelReps
        ).redirectErrorStream(true);
        builder.environment().put("JLC_NATIVE_LOOP_ORDER", variant);
        Process process = builder.start();

        Map<String, String> diag = new LinkedHashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int idx = line.indexOf('=');
                if (idx > 0) {
                    diag.put(line.substring(0, idx), line.substring(idx + 1));
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("Native GEMM deep dive exited with code " + exit + " for size " + size);
        }
        return new Result(size, variant, diag);
    }

    private static String renderMarkdown(List<Result> results, int threads, int microkernelReps) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Native GEMM Deep Dive\n\n");
        sb.append("Configuration:\n\n");
        sb.append("- Threads: `").append(threads).append("`\n");
        sb.append("- Microkernel repetitions: `").append(microkernelReps).append("`\n");
        sb.append("- Scope: packed microkernel only, pack-only traversal, and full blocked GEMM\n\n");

        if (!results.isEmpty()) {
            Result first = results.get(0);
            sb.append("Environment:\n\n");
            sb.append("- Compiled AVX2: `").append(first.diag.getOrDefault("compiled_avx2", "unknown")).append("`\n");
            sb.append("- Compiled FMA: `").append(first.diag.getOrDefault("compiled_fma", "unknown")).append("`\n");
            sb.append("- Runtime AVX2: `").append(first.diag.getOrDefault("runtime_avx2", "unknown")).append("`\n");
            sb.append("- Runtime FMA: `").append(first.diag.getOrDefault("runtime_fma", "unknown")).append("`\n");
            sb.append("- AVX2 enabled: `").append(first.diag.getOrDefault("avx2_enabled", "unknown")).append("`\n\n");
        }

        sb.append("| Size | Variant | MC | KC | NC | MR/NR | Panel-walk GFLOPs | Full GFLOPs | Full ms | JC ms | PC ms | IC ms | Edge ms | Pack share | Kernel share |\n");
        sb.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for (Result result : results) {
            sb.append(String.format(Locale.ROOT,
                "| %d | %s | %s | %s | %s | %s/%s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1f%% | %.1f%% |%n",
                result.size,
                result.variant,
                result.diag.getOrDefault("mc", "?"),
                result.diag.getOrDefault("kc", "?"),
                result.diag.getOrDefault("nc", "?"),
                result.diag.getOrDefault("mr", "?"),
                result.diag.getOrDefault("nr", "?"),
                result.doubleValue("panel_walk_gflops"),
                result.doubleValue("full_gflops"),
                result.doubleValue("full_wall_ms"),
                result.ms("jc_loop_ns"),
                result.ms("pc_loop_ns"),
                result.ms("ic_loop_ns"),
                result.ms("edge_ns"),
                pct(result.longValue("full_pack_a_ns") + result.longValue("full_pack_b_ns"), result.longValue("full_wall_ns")),
                pct(result.longValue("full_kernel_ns"), result.longValue("full_wall_ns"))
            ));
        }
        sb.append('\n');

        sb.append("## Read\n\n");
        for (int size : results.stream().mapToInt(Result::size).distinct().sorted().toArray()) {
            List<Result> perSize = results.stream().filter(r -> r.size == size).toList();
            if (perSize.isEmpty()) {
                continue;
            }
            sb.append("- `").append(size).append("x").append(size).append("`: ");
            Result best = perSize.get(0);
            for (Result candidate : perSize) {
                if (candidate.doubleValue("full_gflops") > best.doubleValue("full_gflops")) {
                    best = candidate;
                }
            }
            sb.append("best current loop-order variant is `").append(best.variant).append("` at ")
                .append(String.format(Locale.ROOT, "%.3f", best.doubleValue("full_gflops"))).append(" GFLOPs");
            sb.append(String.format(Locale.ROOT,
                ", with panel-walk %.3f GFLOPs, JC %.3f ms, PC %.3f ms, IC %.3f ms.%n",
                best.doubleValue("panel_walk_gflops"), best.ms("jc_loop_ns"), best.ms("pc_loop_ns"), best.ms("ic_loop_ns")));
        }

        return sb.toString();
    }

    private static double pct(long numer, long denom) {
        return denom == 0L ? 0.0 : numer * 100.0 / denom;
    }

    private static int[] parseIntList(String raw) {
        String[] parts = raw.split(",");
        int[] values = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Integer.parseInt(parts[i].trim());
        }
        return values;
    }

    private record Result(int size, String variant, Map<String, String> diag) {
        double doubleValue(String key) {
            return Double.parseDouble(diag.getOrDefault(key, "0"));
        }

        long longValue(String key) {
            return Long.parseLong(diag.getOrDefault(key, "0"));
        }

        double ms(String key) {
            return longValue(key) / 1e6;
        }
    }
}
