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
 * Compares native GEMM microkernel candidates in isolation and in the full blocked path.
 */
public final class NativeGemmMicrokernelShootoutRunner {
    private static final List<String> DEFAULT_KERNELS = List.of("avx2_5x4", "avx2_4x4", "avx2_4x8", "scalar");

    private NativeGemmMicrokernelShootoutRunner() {
    }

    public static void main(String[] args) throws Exception {
        int rows = 512;
        int inner = 512;
        int cols = 512;
        int threads = 1;
        int microkernelReps = 250_000;
        List<String> kernels = DEFAULT_KERNELS;

        for (String arg : args) {
            if (arg.startsWith("--rows=")) {
                rows = Integer.parseInt(arg.substring("--rows=".length()));
            } else if (arg.startsWith("--inner=")) {
                inner = Integer.parseInt(arg.substring("--inner=".length()));
            } else if (arg.startsWith("--cols=")) {
                cols = Integer.parseInt(arg.substring("--cols=".length()));
            } else if (arg.startsWith("--threads=")) {
                threads = Integer.parseInt(arg.substring("--threads=".length()));
            } else if (arg.startsWith("--microkernelReps=")) {
                microkernelReps = Integer.parseInt(arg.substring("--microkernelReps=".length()));
            } else if (arg.startsWith("--kernels=")) {
                kernels = List.of(arg.substring("--kernels=".length()).split(","));
            }
        }

        String executable = System.getProperty("jlc.native.gemm.deep.dive.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.gemm.deep.dive.path");
        }

        List<Result> results = new ArrayList<>();
        for (String kernel : kernels) {
            results.add(runCase(executable, rows, inner, cols, threads, microkernelReps, kernel.trim()));
        }

        System.out.println("NATIVE_GEMM_MICROKERNEL_SHOOTOUT");
        System.out.printf(Locale.ROOT, "shape=%dx%d * %dx%d threads=%d microkernelReps=%d%n",
            rows, inner, inner, cols, threads, microkernelReps);
        System.out.println();
        System.out.println("| Kernel | MR/NR | AVX2 | Microkernel GFLOPs | Full GFLOPs | Full ms | Pack share | Kernel share |");
        System.out.println("|---|---|---|---:|---:|---:|---:|---:|");
        for (Result result : results) {
            System.out.printf(Locale.ROOT, "| %s | %s/%s | %s | %.3f | %.3f | %.3f | %.1f%% | %.1f%% |%n",
                result.kernel,
                result.diag.getOrDefault("mr", "?"),
                result.diag.getOrDefault("nr", "?"),
                result.diag.getOrDefault("avx2_enabled", "false"),
                result.doubleValue("microkernel_gflops"),
                result.doubleValue("full_gflops"),
                result.doubleValue("full_wall_ms"),
                pct(result.longValue("full_pack_a_ns") + result.longValue("full_pack_b_ns"), result.longValue("full_wall_ns")),
                pct(result.longValue("full_kernel_ns"), result.longValue("full_wall_ns")));
        }

        Result bestMicro = null;
        Result bestFull = null;
        for (Result result : results) {
            if (bestMicro == null || result.doubleValue("microkernel_gflops") > bestMicro.doubleValue("microkernel_gflops")) {
                bestMicro = result;
            }
            if (bestFull == null || result.doubleValue("full_gflops") > bestFull.doubleValue("full_gflops")) {
                bestFull = result;
            }
        }
        if (bestMicro != null && bestFull != null) {
            System.out.println();
            System.out.printf(Locale.ROOT, "best microkernel: %s at %.3f GFLOPs%n",
                bestMicro.kernel, bestMicro.doubleValue("microkernel_gflops"));
            System.out.printf(Locale.ROOT, "best full path: %s at %.3f GFLOPs%n",
                bestFull.kernel, bestFull.doubleValue("full_gflops"));
        }
    }

    private static Result runCase(String executable, int rows, int inner, int cols,
                                  int threads, int microkernelReps, String kernel) throws Exception {
        ProcessBuilder builder = new ProcessBuilder(
            executable,
            "--rows=" + rows,
            "--inner=" + inner,
            "--cols=" + cols,
            "--threads=" + threads,
            "--microkernel-reps=" + microkernelReps
        ).redirectErrorStream(true);
        builder.environment().put("JLC_NATIVE_MICROKERNEL", kernel);
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
            throw new IllegalStateException("Native GEMM shootout failed for " + kernel + System.lineSeparator() + output);
        }
        return new Result(kernel, diag);
    }

    private static double pct(long numer, long denom) {
        return denom == 0L ? 0.0 : numer * 100.0 / denom;
    }

    private record Result(String kernel, Map<String, String> diag) {
        double doubleValue(String key) {
            return Double.parseDouble(diag.getOrDefault(key, "0"));
        }

        long longValue(String key) {
            return Long.parseLong(diag.getOrDefault(key, "0"));
        }
    }
}
