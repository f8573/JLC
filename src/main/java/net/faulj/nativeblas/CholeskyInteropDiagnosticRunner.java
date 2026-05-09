package net.faulj.nativeblas;

import net.faulj.decomposition.cholesky.CholeskyDecomposition;
import net.faulj.decomposition.result.CholeskyResult;
import net.faulj.matrix.Matrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Focused smoke diagnostic for Java, JNI, and standalone native Cholesky interop.
 */
public final class CholeskyInteropDiagnosticRunner {
    private static final double[] SMOKE3 = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0
    };

    private static final double[] EXPECTED_LOWER = {
        2.0, 0.0, 0.0,
        6.0, 1.0, 0.0,
        -8.0, 5.0, 3.0
    };

    private CholeskyInteropDiagnosticRunner() {
    }

    public static void main(String[] args) throws Exception {
        NativeContext context = new NativeBackend(new JavaBackend()).probe(true);
        if (!context.isAvailable()) {
            throw new IllegalStateException("Native backend unavailable: " + context.getMessage());
        }

        System.out.println("CHOLESKY_INTEROP_DIAGNOSTIC");
        printJavaResult(runJavaSmoke());
        printJniResult(runJniSmoke());
        printCppResult(runCppSmoke());
    }

    private static JavaResult runJavaSmoke() {
        double[] input = SMOKE3.clone();
        double before = checksum(input);
        String previousBackend = System.getProperty("jlc.backend");
        System.setProperty("jlc.backend", "java");
        try {
            CholeskyResult result = new CholeskyDecomposition().decompose(Matrix.wrap(input, 3, 3));
            double[] factor = result.getL().getRawData().clone();
            return new JavaResult(before, checksum(factor), factor, detectTriangle(factor, 3), residual(SMOKE3, factor, 3), firstMismatch(factor));
        } finally {
            if (previousBackend == null) {
                System.clearProperty("jlc.backend");
            } else {
                System.setProperty("jlc.backend", previousBackend);
            }
        }
    }

    private static JniResult runJniSmoke() {
        double[] input = SMOKE3.clone();
        double before = checksum(input);
        int info = NativeBindings.nativeCholeskyDecompose(input, 3);
        return new JniResult(before, checksum(input), info, input.clone(), detectTriangle(input, 3), residual(SMOKE3, input, 3), firstMismatch(input));
    }

    private static CppResult runCppSmoke() throws Exception {
        String executable = System.getProperty("jlc.native.algorithm.bench.path");
        if (executable == null || executable.isBlank()) {
            throw new IllegalStateException("Missing jlc.native.algorithm.bench.path");
        }

        Process process = new ProcessBuilder(
            executable,
            "--algorithm=cholesky",
            "--matrix=smoke3",
            "--diag",
            "--threads=1"
        ).redirectErrorStream(true).start();

        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }
        int exit = process.waitFor();
        if (exit != 0) {
            throw new IllegalStateException("Native diagnostic exited with code " + exit + "\n" + String.join("\n", lines));
        }

        Map<String, String> kv = new LinkedHashMap<>();
        double[] factor = new double[9];
        for (String line : lines) {
            int idx = line.indexOf('=');
            if (idx <= 0) {
                continue;
            }
            String key = line.substring(0, idx);
            String value = line.substring(idx + 1);
            kv.put(key, value);
            if (key.startsWith("block_row_")) {
                int row = Integer.parseInt(key.substring("block_row_".length()));
                String[] parts = value.split(",");
                for (int col = 0; col < Math.min(parts.length, 3); col++) {
                    factor[row * 3 + col] = Double.parseDouble(parts[col]);
                }
            }
        }
        return new CppResult(
            Double.parseDouble(kv.getOrDefault("input_checksum_before", "NaN")),
            Double.parseDouble(kv.getOrDefault("input_checksum_after", "NaN")),
            Integer.parseInt(kv.getOrDefault("status", "-1")),
            Integer.parseInt(kv.getOrDefault("info", "-1")),
            kv.getOrDefault("triangle", "unknown"),
            Double.parseDouble(kv.getOrDefault("residual", "NaN")),
            factor,
            firstMismatch(factor)
        );
    }

    private static void printJavaResult(JavaResult result) {
        System.out.println("java.backend=java");
        System.out.printf(Locale.ROOT, "java.input_checksum_before=%.12f%n", result.inputChecksumBefore());
        System.out.printf(Locale.ROOT, "java.input_checksum_after=%.12f%n", result.inputChecksumAfter());
        System.out.println("java.layout=row-major");
        System.out.println("java.triangle=" + result.triangle());
        System.out.printf(Locale.ROOT, "java.residual=%.12e%n", result.residual());
        System.out.println("java.first_mismatch=" + result.firstMismatch());
        printFactor("java", result.factor());
    }

    private static void printJniResult(JniResult result) {
        System.out.println("jni.backend=builtin");
        System.out.printf(Locale.ROOT, "jni.input_checksum_before=%.12f%n", result.inputChecksumBefore());
        System.out.printf(Locale.ROOT, "jni.input_checksum_after=%.12f%n", result.inputChecksumAfter());
        System.out.println("jni.returned_info=" + result.info());
        System.out.println("jni.layout=row-major");
        System.out.println("jni.triangle=" + result.triangle());
        System.out.printf(Locale.ROOT, "jni.residual=%.12e%n", result.residual());
        System.out.println("jni.first_mismatch=" + result.firstMismatch());
        printFactor("jni", result.factor());
    }

    private static void printCppResult(CppResult result) {
        System.out.println("cpp.backend=standalone");
        System.out.printf(Locale.ROOT, "cpp.input_checksum_before=%.12f%n", result.inputChecksumBefore());
        System.out.printf(Locale.ROOT, "cpp.input_checksum_after=%.12f%n", result.inputChecksumAfter());
        System.out.println("cpp.returned_status=" + result.status());
        System.out.println("cpp.returned_info=" + result.info());
        System.out.println("cpp.layout=row-major");
        System.out.println("cpp.triangle=" + result.triangle());
        System.out.printf(Locale.ROOT, "cpp.residual=%.12e%n", result.residual());
        System.out.println("cpp.first_mismatch=" + result.firstMismatch());
        printFactor("cpp", result.factor());
    }

    private static void printFactor(String prefix, double[] factor) {
        for (int row = 0; row < 3; row++) {
            System.out.printf(Locale.ROOT, "%s.row%d=%.12f,%.12f,%.12f%n",
                prefix, row, factor[row * 3], factor[row * 3 + 1], factor[row * 3 + 2]);
        }
    }

    private static double checksum(double[] matrix) {
        double sum = 0.0;
        for (int i = 0; i < matrix.length; i++) {
            sum = Math.fma(matrix[i], i + 1.0, sum);
        }
        return sum;
    }

    private static String detectTriangle(double[] factor, int n) {
        double lowerAbs = 0.0;
        double upperAbs = 0.0;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (row > col) {
                    lowerAbs += Math.abs(factor[row * n + col]);
                } else if (row < col) {
                    upperAbs += Math.abs(factor[row * n + col]);
                }
            }
        }
        return lowerAbs >= upperAbs ? "lower" : "upper";
    }

    private static double residual(double[] original, double[] factor, int n) {
        double[] reconstructed = new double[n * n];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                double sum = 0.0;
                for (int k = 0; k <= Math.min(row, col); k++) {
                    sum = Math.fma(factor[row * n + k], factor[col * n + k], sum);
                }
                reconstructed[row * n + col] = sum;
            }
        }
        double delta = 0.0;
        double base = 0.0;
        for (int i = 0; i < original.length; i++) {
            double diff = original[i] - reconstructed[i];
            delta = Math.fma(diff, diff, delta);
            base = Math.fma(original[i], original[i], base);
        }
        double denom = Math.sqrt(base);
        return denom == 0.0 ? Math.sqrt(delta) : Math.sqrt(delta) / denom;
    }

    private static String firstMismatch(double[] actual) {
        for (int i = 0; i < actual.length; i++) {
            if (Math.abs(actual[i] - EXPECTED_LOWER[i]) > 1e-9) {
                int row = i / 3;
                int col = i % 3;
                return "row=" + row + ",col=" + col + ",expected=" + EXPECTED_LOWER[i] + ",actual=" + actual[i];
            }
        }
        return "none";
    }

    private record JavaResult(double inputChecksumBefore, double inputChecksumAfter, double[] factor,
                              String triangle, double residual, String firstMismatch) {
    }

    private record JniResult(double inputChecksumBefore, double inputChecksumAfter, int info, double[] factor,
                             String triangle, double residual, String firstMismatch) {
    }

    private record CppResult(double inputChecksumBefore, double inputChecksumAfter, int status, int info,
                             String triangle, double residual, double[] factor, String firstMismatch) {
    }
}
