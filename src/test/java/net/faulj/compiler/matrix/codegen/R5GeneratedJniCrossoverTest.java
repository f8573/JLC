package net.faulj.compiler.matrix.codegen;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.kernel.KernelLowerer;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeGeneratedKernelSupport;

import org.junit.Assume;
import org.junit.Test;

/**
 * Opt-in end-to-end crossover measurement. It includes Java execution and one
 * whole-region JNI call for the scalar and fixed AVX2 R4 variants.
 */
public class R5GeneratedJniCrossoverTest {
    private static volatile double sink;

    @Test
    public void benchmarkBackendCrossoverWhenRequested() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jlc.compiler.r5.jniBenchmark"));
        Assume.assumeTrue(RuntimeCpuFeatures.avx2Supported());
        Assume.assumeTrue(commandWorks("cmake", "--version"));

        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path temporary = Files.createTempDirectory("jlc-r5-jni-");
        String previousLibrary = System.getProperty("jlc.native.lib.path");
        try {
            Path sourceDirectory = temporary.resolve("generated");
            Path buildDirectory = temporary.resolve("native-build");
            Files.createDirectories(sourceDirectory);
            List<Integer> sizes = parseList(
                "jlc.compiler.r5.jniBenchmark.sizes", "1,4,16,64,256");
            List<Integer> operations = parseList(
                "jlc.compiler.r5.jniBenchmark.operations", "2,10,25");
            List<Workload> workloads = new ArrayList<>();
            for (int size : sizes) {
                for (int arithmeticOps : operations) {
                    Workload workload = workload(size, arithmeticOps);
                    workloads.add(workload);
                    for (GeneratedKernelSource source : workload.sources()) {
                        Files.writeString(sourceDirectory.resolve(source.symbol() + ".cpp"),
                            source.source(), StandardCharsets.UTF_8);
                    }
                }
            }

            configureAndBuild(repository, buildDirectory, sourceDirectory);
            Path library = buildDirectory.resolve("lib")
                .resolve(System.mapLibraryName("jlc_native"));
            assertTrue("custom generated JNI library was not built", Files.isRegularFile(library));
            System.setProperty("jlc.native.lib.path", library.toString());
            assertTrue("custom generated JNI library did not load",
                NativeGeneratedKernelSupport.isAvailable());

            Path report = repository.resolve("build/reports/r5/jni-crossover.txt");
            Files.createDirectories(report.getParent());
            StringBuilder output = new StringBuilder(
                "# R5 end-to-end JNI crossover\n"
                    + "# warmup=10 samples=11 median/min/max/MAD; includes JNI pin/lookup\n"
                    + "| Kernel | Shape | R2 Java ns | Scalar JNI ns | AVX2 JNI ns | Selected |\n"
                    + "|---|---:|---:|---:|---:|---|\n");
            for (Workload workload : workloads) {
                Matrix expected = workload.program().execute();
                GeneratedKernelSource scalar = source(workload, CodegenBackend.SCALAR_CPP);
                GeneratedKernelSource avx2 = source(workload, CodegenBackend.AVX2);
                KernelBenchmarkStatistics javaStats = measureJava(workload.program(),
                    expected.getRawData());
                KernelBenchmarkStatistics scalarStats = measureNative(workload, scalar,
                    expected.getRawData());
                KernelBenchmarkStatistics avx2Stats = measureNative(workload, avx2,
                    expected.getRawData());
                String selected = selected(javaStats, scalarStats, avx2Stats);
                String line = "| " + workload.operations() + "-op | "
                    + workload.size() + "x" + workload.size() + " | "
                    + format(javaStats) + " | " + format(scalarStats) + " | "
                    + format(avx2Stats) + " | " + selected + " |\n";
                output.append(line);
                System.out.print(line);
            }
            Files.writeString(report, output.toString(), StandardCharsets.UTF_8);
        } finally {
            if (previousLibrary == null) {
                System.clearProperty("jlc.native.lib.path");
            } else {
                System.setProperty("jlc.native.lib.path", previousLibrary);
            }
            try (Stream<Path> paths = Files.walk(temporary)) {
                paths.sorted((left, right) -> right.getNameCount() - left.getNameCount())
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException ignored) {
                        }
                    });
            }
        }
    }

    private static Workload workload(int size, int operations) {
        Matrix a = matrix(size, size, 0.125);
        Matrix b = matrix(size, size, 0.625);
        MatrixExpr expression = MatrixExpr.input(a);
        for (int operation = 0; operation < operations; operation++) {
            expression = (operation & 1) == 0
                ? expression.scale(1.001 + operation * 0.0001)
                : expression.add(MatrixExpr.input(b));
        }
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        KernelLoweringResult lowering = KernelLowerer.lower(
            program.cpuPlan().fusedRegions().get(0));
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
        List<GeneratedKernelSource> sources = KernelVariantGenerator.enumerate(plan).stream()
            .filter(KernelVariantCandidate::eligible)
            .map(candidate -> KernelCodeGenerator.variant(
                plan, candidate.signature(), CppEmissionOptions.nativeRegistry()))
            .toList();
        return new Workload(size, operations, program, plan, sources);
    }

    private static GeneratedKernelSource source(Workload workload, CodegenBackend backend) {
        KernelVariantSignature signature = backend == CodegenBackend.AVX2
            ? KernelVariantSignature.baselineAvx2(workload.plan().signature())
            : KernelVariantSignature.legacyScalar(workload.plan().signature());
        return workload.sources().stream()
            .filter(candidate -> candidate.variantSignature().equals(signature))
            .findFirst()
            .orElseThrow(() -> new AssertionError("missing crossover source " + signature));
    }

    private static KernelBenchmarkStatistics measureJava(CompiledMatrixProgram program,
                                                         double[] expected) {
        for (int warmup = 0; warmup < 10; warmup++) {
            Matrix actual = program.execute();
            assertNear(expected, actual.getRawData());
            sink += actual.getRawData()[0];
        }
        long[] samples = new long[11];
        for (int sample = 0; sample < samples.length; sample++) {
            long start = System.nanoTime();
            Matrix actual = program.execute();
            samples[sample] = Math.max(1L, System.nanoTime() - start);
            assertNear(expected, actual.getRawData());
            sink += actual.getRawData()[sample % actual.getRawData().length];
        }
        return KernelBenchmarkStatistics.from(samples);
    }

    private static KernelBenchmarkStatistics measureNative(Workload workload,
                                                            GeneratedKernelSource source,
                                                            double[] expected) {
        double[][] inputs = {workload.input(0), workload.input(1)};
        double[] output = new double[expected.length];
        String signature = source.signature().canonicalText();
        String variant = source.variantSignature().canonicalText();
        for (int warmup = 0; warmup < 10; warmup++) {
            assertTrue(NativeGeneratedKernelSupport.executeVariant(
                signature, variant, inputs, output, workload.size(), workload.size()));
            assertNear(expected, output);
            sink += output[workload.safeIndex()];
        }
        long[] samples = new long[11];
        for (int sample = 0; sample < samples.length; sample++) {
            long start = System.nanoTime();
            assertTrue(NativeGeneratedKernelSupport.executeVariant(
                signature, variant, inputs, output, workload.size(), workload.size()));
            samples[sample] = Math.max(1L, System.nanoTime() - start);
            assertNear(expected, output);
            sink += output[sample % Math.max(1, output.length)];
        }
        return KernelBenchmarkStatistics.from(samples);
    }

    private static String selected(KernelBenchmarkStatistics javaStats,
                                   KernelBenchmarkStatistics scalarStats,
                                   KernelBenchmarkStatistics avx2Stats) {
        double java = javaStats.medianNanos();
        double scalar = scalarStats.medianNanos();
        double avx2 = avx2Stats.medianNanos();
        if (java <= scalar && java <= avx2) return "R2_JAVA";
        if (scalar <= avx2) return "SCALAR_JNI";
        return "AVX2_JNI";
    }

    private static String format(KernelBenchmarkStatistics stats) {
        return String.format("%.1f", stats.medianNanos());
    }

    private static void assertNear(double[] expected, double[] actual) {
        assertTrue("native/Java result mismatch", expected.length == actual.length);
        for (int index = 0; index < expected.length; index++) {
            double scale = Math.max(1.0, Math.max(Math.abs(expected[index]), Math.abs(actual[index])));
            assertTrue("mismatch at " + index,
                Math.abs(expected[index] - actual[index]) <= 1.0e-12 * scale);
        }
    }

    private static void configureAndBuild(Path repository,
                                          Path buildDirectory,
                                          Path sourceDirectory) throws Exception {
        run(List.of(
            "cmake", "-S", repository.resolve("native-backend").toString(),
            "-B", buildDirectory.toString(),
            "-DJLC_JAVA_HOME=" + Path.of(System.getProperty("java.home")).toAbsolutePath(),
            "-DJLC_NATIVE_ENABLE_VENDOR_BLAS=OFF",
            "-DJLC_NATIVE_ENABLE_MARCH_NATIVE=OFF",
            "-DJLC_R4_GENERATED_SOURCE_DIR=" + sourceDirectory),
            repository, 60);
        run(List.of("cmake", "--build", buildDirectory.toString(),
            "--target", "jlc_native", "-j2"), repository, 180);
    }

    private static void run(List<String> command, Path directory, long timeoutSeconds)
        throws Exception {
        Process process = new ProcessBuilder(new ArrayList<>(command))
            .directory(directory.toFile()).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue("command failed: " + command + "\n" + output,
            process.waitFor(timeoutSeconds, TimeUnit.SECONDS) && process.exitValue() == 0);
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + (index % 97) * 0.001;
        }
        return Matrix.wrap(data, rows, columns);
    }

    private static List<Integer> parseList(String property, String fallback) {
        return Arrays.stream(System.getProperty(property, fallback).split(","))
            .map(String::trim).filter(value -> !value.isEmpty()).map(Integer::parseInt).toList();
    }

    private static boolean commandWorks(String command, String argument) {
        try {
            Process process = new ProcessBuilder(command, argument)
                .redirectErrorStream(true).start();
            return process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException | InterruptedException failure) {
            if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
            return false;
        }
    }

    private record Workload(int size,
                            int operations,
                            CompiledMatrixProgram program,
                            PseudokernelPlan plan,
                            List<GeneratedKernelSource> sources) {
        private double[] input(int index) {
            Matrix matrix = index == 0
                ? program.cpuPlan().inputBindings().get(0).matrix()
                : program.cpuPlan().inputBindings().get(1).matrix();
            return matrix.getRawData();
        }

        private int safeIndex() {
            return Math.max(0, size * size - 1);
        }
    }
}
