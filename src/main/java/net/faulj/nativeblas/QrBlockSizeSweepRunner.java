package net.faulj.nativeblas;

import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.QRResult;
import net.faulj.matrix.Matrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public final class QrBlockSizeSweepRunner {
    private QrBlockSizeSweepRunner() {
    }

    public static void main(String[] args) throws IOException {
        int[] blockSizes = {16, 32, 48, 64, 96};
        Shape[] shapes = {
            new Shape("square", 512, 512),
            new Shape("square", 1024, 1024),
            new Shape("tall", 1024, 256),
            new Shape("tall", 2048, 512),
            new Shape("wide", 256, 1024),
            new Shape("wide", 512, 2048)
        };
        int warmup = 1;
        int runs = 2;
        Mode[] modes = Mode.values();
        int[] threads = {1};
        Path out = Path.of("build", "reports", "qr_block_size_sweep.csv");

        for (String arg : args) {
            if (arg.startsWith("--blockSizes=")) {
                blockSizes = parseInts(arg.substring("--blockSizes=".length()), blockSizes);
            } else if (arg.startsWith("--blocks=")) {
                blockSizes = parseInts(arg.substring("--blocks=".length()), blockSizes);
            } else if (arg.startsWith("--sizes=")) {
                shapes = squareShapes(parseInts(arg.substring("--sizes=".length()), extractSquareSizes(shapes)));
            } else if (arg.startsWith("--shapes=")) {
                shapes = parseShapes(arg.substring("--shapes=".length()), shapes);
            } else if (arg.startsWith("--mode=")) {
                modes = parseModes(arg.substring("--mode=".length()), modes);
            } else if (arg.startsWith("--threads=")) {
                threads = parseInts(arg.substring("--threads=".length()), threads);
            } else if (arg.startsWith("--warmup=")) {
                warmup = Math.max(0, Integer.parseInt(arg.substring("--warmup=".length())));
            } else if (arg.startsWith("--runs=")) {
                runs = Math.max(1, Integer.parseInt(arg.substring("--runs=".length())));
            } else if (arg.startsWith("--out=")) {
                out = Path.of(arg.substring("--out=".length()));
            }
        }

        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!"native".equals(snapshot.activeBackend())) {
            throw new IllegalStateException("Native backend unavailable: " + snapshot.nativeContext().getMessage());
        }

        double gemmAnchorGflops = measureGemmAnchorGflops();
        List<String> lines = new ArrayList<>();
        lines.add(String.join(",",
            "mode", "family", "rows", "cols", "blockSize", "threads", "backendWinner",
            "javaBestMs", "nativeProductionMs", "nativeHarnessMs", "nativeGflops", "pesVsGemm",
            "residual", "orthogonality",
            "javaNativeBoundaryNs", "validationReconstructNs", "validationResidualNs", "validationOrthogonalityNs",
            "qrWallNs", "qrFactorizeNs", "qrInputTransposeNs", "qrPanelNs", "qrReflectorPackNs", "qrTBuildNs",
            "qrTrailingPackNs", "qrTrailingUnpackNs", "qrTrailingGemmNs", "qrTrailingTApplyNs",
            "qrRExtractNs", "qrQInitNs", "qrQBuildNs", "qrQGemmNs", "qrQTApplyNs",
            "gemmCalls", "gemmWallNs", "gemmPackANs", "gemmPackBNs", "gemmKernelNs",
            "gemmPackABytes", "gemmPackBBytes", "gemmMicrotileCalls", "gemmMr", "gemmNr", "gemmMc", "gemmNc", "gemmKc"
        ));
        Files.createDirectories(out.toAbsolutePath().getParent());
        Files.write(out, lines);

        System.out.printf(Locale.ROOT, "qr_sweep_gemm_anchor_gflops=%.6f%n", gemmAnchorGflops);
        for (int threadCount : threads) {
            setNativeQrGemmThreads(threadCount);
            for (int blockSize : blockSizes) {
                setNativeQrBlockSize(blockSize);
                for (Shape shape : shapes) {
                    Matrix input = randomMatrix(shape.rows, shape.cols, 17_000L + shape.rows * 31L + shape.cols);
                    for (Mode mode : modes) {
                    Result javaResult = measureJava(input, mode, warmup, runs);
                    Result nativeResult = measureNative(input, mode, warmup, runs);
                    String winner = nativeResult.productionSeconds <= javaResult.productionSeconds ? "cpp" : "java";
                    double flops = qrFlops(shape.rows, shape.cols, mode);
                    double nativeGflops = flops / nativeResult.productionSeconds / 1e9;
                    double pes = gemmAnchorGflops > 0.0 ? nativeGflops / gemmAnchorGflops : 0.0;

                    lines.add(csv(
                        mode.cli, shape.family, shape.rows, shape.cols, blockSize, threadCount, winner,
                        javaResult.productionSeconds * 1000.0, nativeResult.productionSeconds * 1000.0,
                        nativeResult.harnessSeconds * 1000.0,
                        nativeGflops, pes, nativeResult.residual, nativeResult.orthogonality,
                        nativeResult.javaNativeBoundaryNanos(), nativeResult.validationReconstructNanos,
                        nativeResult.validationResidualNanos, nativeResult.validationOrthogonalityNanos,
                        nativeResult.qrProfile.wallNanos(), nativeResult.qrProfile.factorizeNanos(),
                        nativeResult.qrProfile.inputTransposeNanos(), nativeResult.qrProfile.panelNanos(),
                        nativeResult.qrProfile.reflectorPackNanos(),
                        nativeResult.qrProfile.tBuildNanos(), nativeResult.qrProfile.trailingPackNanos(),
                        nativeResult.qrProfile.trailingUnpackNanos(), nativeResult.qrProfile.trailingGemmNanos(),
                        nativeResult.qrProfile.trailingTApplyNanos(), nativeResult.qrProfile.rExtractNanos(),
                        nativeResult.qrProfile.qInitNanos(), nativeResult.qrProfile.qBuildNanos(),
                        nativeResult.qrProfile.qGemmNanos(),
                        nativeResult.qrProfile.qTApplyNanos(), nativeResult.gemmProfile.calls(),
                        nativeResult.gemmProfile.wallNanos(), nativeResult.gemmProfile.packANanos(),
                        nativeResult.gemmProfile.packBNanos(), nativeResult.gemmProfile.kernelNanos(),
                        nativeResult.gemmProfile.packABytes(), nativeResult.gemmProfile.packBBytes(),
                        nativeResult.gemmProfile.microtileCalls(), nativeResult.gemmProfile.lastMr(),
                        nativeResult.gemmProfile.lastNr(), nativeResult.gemmProfile.lastMc(),
                        nativeResult.gemmProfile.lastNc(), nativeResult.gemmProfile.lastKc()
                    ));
                    Files.write(out, lines);

                    System.out.printf(Locale.ROOT,
                        "mode=%s shape=%s/%dx%d bs=%d threads=%d winner=%s nativeProd=%.3fms harness=%.3fms %.3fgflops pes=%.3f residual=%.3e orth=%.3e%n",
                        mode.cli, shape.family, shape.rows, shape.cols, blockSize, threadCount, winner,
                        nativeResult.productionSeconds * 1000.0, nativeResult.harnessSeconds * 1000.0, nativeGflops, pes,
                        nativeResult.residual, nativeResult.orthogonality);
                    }
                }
            }
        }

        Files.write(out, lines);
        System.out.println("qr_block_size_sweep_written=" + out.toAbsolutePath());
    }

    private static Result measureJava(Matrix input, Mode mode, int warmup, int runs) {
        for (int i = 0; i < warmup; i++) {
            executeJava(input, mode);
        }
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            executeJava(input, mode);
            best = Math.min(best, (System.nanoTime() - start) / 1e9);
        }
        return new Result(best, best, Double.NaN, Double.NaN, 0L, 0L, 0L, NativeQrProfile.EMPTY, NativeGemmProfile.EMPTY);
    }

    private static Result measureNative(Matrix input, Mode mode, int warmup, int runs) {
        for (int i = 0; i < warmup; i++) {
            executeNative(input, mode);
        }
        double best = Double.POSITIVE_INFINITY;
        double bestHarness = Double.POSITIVE_INFINITY;
        double residual = Double.NaN;
        double orthogonality = Double.NaN;
        long reconstructionNanos = 0L;
        long residualNanos = 0L;
        long orthogonalityNanos = 0L;
        NativeQrProfile bestQr = NativeQrProfile.EMPTY;
        NativeGemmProfile bestGemm = NativeGemmProfile.EMPTY;
        NativeProfiling.setEnabled(true);
        NativeProfiling.setQrEnabled(true);
        for (int i = 0; i < runs; i++) {
            NativeProfiling.reset();
            NativeProfiling.resetQr();
            long start = System.nanoTime();
            Validation validation = executeNative(input, mode);
            long harnessNanos = System.nanoTime() - start;
            NativeQrProfile currentQr = NativeProfiling.qrSnapshot().orElse(NativeQrProfile.EMPTY);
            NativeGemmProfile currentGemm = NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY);
            double productionSeconds = currentQr.wallNanos() > 0
                ? currentQr.wallNanos() / 1e9
                : validation.nativeCallNanos / 1e9;
            if (productionSeconds < best) {
                best = productionSeconds;
                bestHarness = harnessNanos / 1e9;
                residual = validation.residual;
                orthogonality = validation.orthogonality;
                reconstructionNanos = validation.reconstructionNanos;
                residualNanos = validation.residualNanos;
                orthogonalityNanos = validation.orthogonalityNanos;
                bestQr = currentQr;
                bestGemm = currentGemm;
            }
        }
        NativeProfiling.setEnabled(false);
        NativeProfiling.setQrEnabled(false);
        return new Result(best, bestHarness, residual, orthogonality, reconstructionNanos, residualNanos,
            orthogonalityNanos, bestQr, bestGemm);
    }

    private static void executeJava(Matrix input, Mode mode) {
        Matrix copy = input.copy();
        switch (mode) {
            case FACTORIZE -> HouseholderQR.factorize(copy);
            case THIN -> HouseholderQR.decomposeThin(copy);
            case FULL -> HouseholderQR.decompose(copy);
        }
    }

    private static Validation executeNative(Matrix input, Mode mode) {
        int rows = input.getRowCount();
        int cols = input.getColumnCount();
        double[] a = input.getRawData().clone();
        if (mode == Mode.FACTORIZE) {
            long nativeStart = System.nanoTime();
            NativeBindings.nativeQrFactorizeOnly(a, rows, cols);
            return new Validation(Double.NaN, Double.NaN, System.nanoTime() - nativeStart, 0L, 0L, 0L);
        }
        int qCols = mode == Mode.THIN ? Math.min(rows, cols) : rows;
        double[] q = new double[rows * qCols];
        double[] r = new double[qCols * cols];
        long nativeStart = System.nanoTime();
        NativeBindings.nativeQrDecompose(a, rows, cols, qCols, q, r);
        long nativeNanos = System.nanoTime() - nativeStart;

        long reconstructionStart = System.nanoTime();
        double[] reconstructed = reconstruct(q, rows, qCols, r, cols);
        long reconstructionNanos = System.nanoTime() - reconstructionStart;

        long residualStart = System.nanoTime();
        double residual = residualError(input.getRawData(), reconstructed);
        long residualNanos = System.nanoTime() - residualStart;

        long orthogonalityStart = System.nanoTime();
        double orthogonality = orthogonalityError(q, rows, qCols);
        long orthogonalityNanos = System.nanoTime() - orthogonalityStart;

        return new Validation(residual, orthogonality, nativeNanos, reconstructionNanos, residualNanos, orthogonalityNanos);
    }

    private static double measureGemmAnchorGflops() {
        int n = 1024;
        double[] a = randomArray(n * n, 101L);
        double[] b = randomArray(n * n, 102L);
        double[] c = new double[n * n];
        NativeBindings.nativeGemm(a, n, n, b, n, n, c, n, n, 1.0, 0.0,
            1, 0);
        long start = System.nanoTime();
        NativeBindings.nativeGemm(a, n, n, b, n, n, c, n, n, 1.0, 0.0,
            1, 0);
        double seconds = (System.nanoTime() - start) / 1e9;
        return 2.0 * n * (double) n * n / seconds / 1e9;
    }

    private static void setNativeQrBlockSize(int blockSize) {
        NativeBindings.nativeQrSetBlockSizeOverride(blockSize);
    }

    private static void setNativeQrGemmThreads(int threads) {
        NativeBindings.nativeQrSetGemmThreadsOverride(threads);
    }

    private static Matrix randomMatrix(int rows, int cols, long seed) {
        return Matrix.wrap(randomArray(rows * cols, seed), rows, cols);
    }

    private static double[] randomArray(int length, long seed) {
        Random random = new Random(seed);
        double[] data = new double[length];
        for (int i = 0; i < length; i++) {
            data[i] = random.nextDouble() - 0.5;
        }
        return data;
    }

    private static double qrFlops(int rows, int cols, Mode mode) {
        int k = Math.min(rows, cols);
        double factorize = 2.0 * rows * (double) cols * k - (2.0 / 3.0) * k * (double) k * k;
        if (mode == Mode.FACTORIZE) {
            return factorize;
        }
        int qCols = mode == Mode.THIN ? k : rows;
        return factorize + 2.0 * rows * (double) qCols * k;
    }

    private static double[] reconstruct(double[] q, int rows, int qCols, double[] r, int cols) {
        double[] reconstructed = new double[rows * cols];
        for (int row = 0; row < rows; row++) {
            int outBase = row * cols;
            int qBase = row * qCols;
            for (int col = 0; col < cols; col++) {
                double value = 0.0;
                for (int mid = 0; mid < qCols; mid++) {
                    value = Math.fma(q[qBase + mid], r[mid * cols + col], value);
                }
                reconstructed[outBase + col] = value;
            }
        }
        return reconstructed;
    }

    private static double residualError(double[] a, double[] reconstructed) {
        double diff = 0.0;
        double norm = 0.0;
        int limit = Math.min(a.length, reconstructed.length);
        for (int i = 0; i < limit; i++) {
            double delta = a[i] - reconstructed[i];
            diff = Math.fma(delta, delta, diff);
            norm = Math.fma(a[i], a[i], norm);
        }
        return Math.sqrt(diff) / Math.max(1.0, Math.sqrt(norm));
    }

    private static double orthogonalityError(double[] q, int rows, int cols) {
        double diff = 0.0;
        for (int left = 0; left < cols; left++) {
            for (int right = 0; right < cols; right++) {
                double dot = 0.0;
                for (int row = 0; row < rows; row++) {
                    dot = Math.fma(q[row * cols + left], q[row * cols + right], dot);
                }
                double expected = left == right ? 1.0 : 0.0;
                double delta = dot - expected;
                diff = Math.fma(delta, delta, diff);
            }
        }
        return Math.sqrt(diff);
    }

    private static int[] parseInts(String value, int[] fallback) {
        String[] parts = value.split(",");
        int[] parsed = new int[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) {
                parsed[i] = Integer.parseInt(parts[i].trim());
            }
            return parsed;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private static Shape[] parseShapes(String value, Shape[] fallback) {
        String[] parts = value.split(",");
        List<Shape> shapes = new ArrayList<>();
        try {
            for (String part : parts) {
                String[] dims = part.toLowerCase(Locale.ROOT).split("x");
                int rows = Integer.parseInt(dims[0].trim());
                int cols = Integer.parseInt(dims[1].trim());
                String family = rows == cols ? "square" : rows > cols ? "tall" : "wide";
                shapes.add(new Shape(family, rows, cols));
            }
            return shapes.toArray(Shape[]::new);
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private static int[] extractSquareSizes(Shape[] shapes) {
        int[] out = new int[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            out[i] = Math.max(shapes[i].rows, shapes[i].cols);
        }
        return out;
    }

    private static Shape[] squareShapes(int[] sizes) {
        Shape[] shapes = new Shape[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            shapes[i] = new Shape("square", sizes[i], sizes[i]);
        }
        return shapes;
    }

    private static Mode[] parseModes(String value, Mode[] fallback) {
        try {
            String[] parts = value.split(",");
            List<Mode> modes = new ArrayList<>();
            for (String part : parts) {
                String normalized = part.trim().toLowerCase(Locale.ROOT);
                for (Mode mode : Mode.values()) {
                    if (mode.cli.equals(normalized)) {
                        modes.add(mode);
                    }
                }
            }
            return modes.isEmpty() ? fallback : modes.toArray(Mode[]::new);
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private static String csv(Object... values) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            Object value = values[i];
            if (value instanceof Double d) {
                out.append(String.format(Locale.ROOT, "%.9g", d));
            } else {
                out.append(value);
            }
        }
        return out.toString();
    }

    private enum Mode {
        FACTORIZE("factorize"),
        THIN("thin"),
        FULL("full");

        final String cli;

        Mode(String cli) {
            this.cli = cli;
        }
    }

    private record Shape(String family, int rows, int cols) {
    }

    private record Validation(
        double residual,
        double orthogonality,
        long nativeCallNanos,
        long reconstructionNanos,
        long residualNanos,
        long orthogonalityNanos
    ) {
    }

    private record Result(
        double productionSeconds,
        double harnessSeconds,
        double residual,
        double orthogonality,
        long validationReconstructNanos,
        long validationResidualNanos,
        long validationOrthogonalityNanos,
        NativeQrProfile qrProfile,
        NativeGemmProfile gemmProfile
    ) {
        long javaNativeBoundaryNanos() {
            long elapsedNanos = Math.round(harnessSeconds * 1e9);
            return Math.max(0L, elapsedNanos - qrProfile.wallNanos());
        }
    }
}
