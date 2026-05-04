package net.faulj.nativeblas;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public final class QrTrailingUpdateGemmBenchmark {
    private QrTrailingUpdateGemmBenchmark() {
    }

    public static void main(String[] args) {
        Case[] cases = {
            new Case(512, 16, 64),
            new Case(512, 32, 96),
            new Case(1024, 16, 64),
            new Case(1024, 32, 96),
            new Case(1536, 32, 96),
            new Case(2048, 32, 96),
            new Case(2048, 64, 128)
        };
        int warmup = 2;
        int runs = 5;
        int threads = 1;

        for (String arg : args) {
            if (arg.startsWith("--cases=")) {
                cases = parseCases(arg.substring("--cases=".length()), cases);
            } else if (arg.startsWith("--warmup=")) {
                warmup = Math.max(0, Integer.parseInt(arg.substring("--warmup=".length())));
            } else if (arg.startsWith("--runs=")) {
                runs = Math.max(1, Integer.parseInt(arg.substring("--runs=".length())));
            } else if (arg.startsWith("--threads=")) {
                threads = Math.max(1, Integer.parseInt(arg.substring("--threads=".length())));
            }
        }

        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!"native".equals(snapshot.activeBackend())) {
            throw new IllegalStateException("Native backend unavailable: " + snapshot.nativeContext().getMessage());
        }

        System.out.println("QR_TRAILING_UPDATE_GEMM_BENCHMARK");
        System.out.println("provider=" + snapshot.nativeContext().getProviderDescription());
        System.out.println("csv,op,activeRows,panelSize,blockCols,provider,bestMs,meanMs,gflops,gemmCalls,gemmWallNs,packANs,packBNs,kernelNs,packABytes,packBBytes,microtileCalls,vendorCalls,mr,nr,mc,nc,kc,maxAbsDiff");

        for (Case testCase : cases) {
            Data data = randomData(testCase, 31_000L + testCase.activeRows * 17L + testCase.panelSize);
            Result builtinVtc = measureVtc(data, testCase, warmup, runs, threads, NativeFlags.FORCE_BUILTIN);
            Result vendorVtc = measureVtc(data, testCase, warmup, runs, threads, NativeFlags.FORCE_VENDOR);
            Result builtinVw = measureVw(data, testCase, warmup, runs, threads, NativeFlags.FORCE_BUILTIN);
            Result vendorVw = measureVw(data, testCase, warmup, runs, threads, NativeFlags.FORCE_VENDOR);
            print("vtc", testCase, "builtin", builtinVtc, 0.0);
            print("vtc", testCase, "vendor", vendorVtc, maxAbsDiff(builtinVtc.output, vendorVtc.output));
            print("vw", testCase, "builtin", builtinVw, 0.0);
            print("vw", testCase, "vendor", vendorVw, maxAbsDiff(builtinVw.output, vendorVw.output));
        }
    }

    private static Result measureVtc(Data data, Case testCase, int warmup, int runs, int threads, int flags) {
        int outputLength = testCase.panelSize * testCase.blockCols;
        for (int i = 0; i < warmup; i++) {
            double[] w = new double[outputLength];
            NativeBindings.nativeGemmStrided(
                data.vtPack, 0, testCase.activeRows, testCase.panelSize, testCase.activeRows, 0,
                data.cBlock, 0, testCase.blockCols, testCase.activeRows, testCase.blockCols, 0,
                w, 0, testCase.blockCols, testCase.panelSize, testCase.blockCols, 0,
                1.0, 0.0, threads, flags
            );
        }
        return measure(outputLength, runs, () -> {
            double[] w = new double[outputLength];
            NativeBindings.nativeGemmStrided(
                data.vtPack, 0, testCase.activeRows, testCase.panelSize, testCase.activeRows, 0,
                data.cBlock, 0, testCase.blockCols, testCase.activeRows, testCase.blockCols, 0,
                w, 0, testCase.blockCols, testCase.panelSize, testCase.blockCols, 0,
                1.0, 0.0, threads, flags
            );
            return w;
        });
    }

    private static Result measureVw(Data data, Case testCase, int warmup, int runs, int threads, int flags) {
        int outputLength = testCase.activeRows * testCase.blockCols;
        for (int i = 0; i < warmup; i++) {
            double[] c = data.cBlock.clone();
            NativeBindings.nativeGemmStrided(
                data.vtPack, 0, testCase.activeRows, testCase.panelSize, testCase.activeRows, NativeFlags.A_TRANSPOSE,
                data.w, 0, testCase.blockCols, testCase.panelSize, testCase.blockCols, 0,
                c, 0, testCase.blockCols, testCase.activeRows, testCase.blockCols, 0,
                -1.0, 1.0, threads, flags
            );
        }
        return measure(outputLength, runs, () -> {
            double[] c = data.cBlock.clone();
            NativeBindings.nativeGemmStrided(
                data.vtPack, 0, testCase.activeRows, testCase.panelSize, testCase.activeRows, NativeFlags.A_TRANSPOSE,
                data.w, 0, testCase.blockCols, testCase.panelSize, testCase.blockCols, 0,
                c, 0, testCase.blockCols, testCase.activeRows, testCase.blockCols, 0,
                -1.0, 1.0, threads, flags
            );
            return c;
        });
    }

    private static Result measure(int outputLength, int runs, Operation operation) {
        NativeProfiling.setEnabled(true);
        double best = Double.POSITIVE_INFINITY;
        double total = 0.0;
        double[] bestOutput = new double[outputLength];
        NativeGemmProfile bestProfile = NativeGemmProfile.EMPTY;
        for (int run = 0; run < runs; run++) {
            NativeProfiling.reset();
            long start = System.nanoTime();
            double[] output = operation.run();
            double seconds = (System.nanoTime() - start) / 1e9;
            total += seconds;
            if (seconds < best) {
                best = seconds;
                bestOutput = output;
                bestProfile = NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY);
            }
        }
        NativeProfiling.setEnabled(false);
        return new Result(best, total / runs, bestOutput, bestProfile);
    }

    private static void print(String op, Case testCase, String provider, Result result, double maxAbsDiff) {
        double flops = 2.0 * testCase.activeRows * (double) testCase.panelSize * testCase.blockCols;
        NativeGemmProfile p = result.profile;
        System.out.printf(Locale.ROOT,
            "csv,%s,%d,%d,%d,%s,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6e%n",
            op, testCase.activeRows, testCase.panelSize, testCase.blockCols, provider,
            result.bestSeconds * 1000.0, result.meanSeconds * 1000.0, flops / result.bestSeconds / 1e9,
            p.calls(), p.wallNanos(), p.packANanos(), p.packBNanos(), p.kernelNanos(),
            p.packABytes(), p.packBBytes(), p.microtileCalls(), p.vendorCalls(),
            p.lastMr(), p.lastNr(), p.lastMc(), p.lastNc(), p.lastKc(), maxAbsDiff
        );
    }

    private static Data randomData(Case testCase, long seed) {
        Random random = new Random(seed);
        return new Data(
            randomArray(random, testCase.panelSize * testCase.activeRows),
            randomArray(random, testCase.activeRows * testCase.blockCols),
            randomArray(random, testCase.panelSize * testCase.blockCols)
        );
    }

    private static double[] randomArray(Random random, int length) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = random.nextDouble() - 0.5;
        }
        return out;
    }

    private static double maxAbsDiff(double[] left, double[] right) {
        double max = 0.0;
        int limit = Math.min(left.length, right.length);
        for (int i = 0; i < limit; i++) {
            max = Math.max(max, Math.abs(left[i] - right[i]));
        }
        return max;
    }

    private static Case[] parseCases(String value, Case[] fallback) {
        try {
            String[] parts = value.split(",");
            List<Case> cases = new ArrayList<>();
            for (String part : parts) {
                String[] dims = part.toLowerCase(Locale.ROOT).trim().split("x");
                if (dims.length == 3) {
                    cases.add(new Case(
                        Integer.parseInt(dims[0].trim()),
                        Integer.parseInt(dims[1].trim()),
                        Integer.parseInt(dims[2].trim())
                    ));
                }
            }
            return cases.isEmpty() ? fallback : cases.toArray(Case[]::new);
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    @FunctionalInterface
    private interface Operation {
        double[] run();
    }

    private record Case(int activeRows, int panelSize, int blockCols) {
    }

    private record Data(double[] vtPack, double[] cBlock, double[] w) {
    }

    private record Result(double bestSeconds, double meanSeconds, double[] output, NativeGemmProfile profile) {
    }
}
