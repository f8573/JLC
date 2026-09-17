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

/** Optional JNI bridge benchmark for one registered generated whole-region call. */
public class R4GeneratedJniBenchmarkTest {
    @Test
    public void benchmarkJniBridgeWhenRequested() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jlc.compiler.r4.jniBenchmark"));
        Assume.assumeTrue(RuntimeCpuFeatures.avx2Supported());
        Assume.assumeTrue(commandWorks("cmake", "--version"));

        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path temporary = Files.createTempDirectory("jlc-r4-jni-");
        try {
            Path sourceDirectory = temporary.resolve("generated");
            Path buildDirectory = temporary.resolve("native-build");
            Files.createDirectories(sourceDirectory);

            Matrix a = matrix(256, 256, 1.0);
            Matrix b = matrix(256, 256, 2.0);
            KernelLoweringResult lowering = lower(
                MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
            GeneratedKernelSource avx2 = KernelCodeGenerator.avx2(
                PseudokernelPlanner.plan(lowering.program().function()),
                CppEmissionOptions.nativeRegistry());
            KernelLoweringResult zeroLowering = lower(
                MatrixExpr.input(new Matrix(0, 3)).scale(2.0)
                    .add(MatrixExpr.input(new Matrix(0, 3))));
            GeneratedKernelSource zeroScalar = KernelCodeGenerator.scalar(
                PseudokernelPlanner.plan(zeroLowering.program().function()),
                CppEmissionOptions.nativeRegistry());
            Files.writeString(sourceDirectory.resolve(avx2.symbol() + ".cpp"),
                avx2.source(), StandardCharsets.UTF_8);
            Files.writeString(sourceDirectory.resolve(zeroScalar.symbol() + ".cpp"),
                zeroScalar.source(), StandardCharsets.UTF_8);

            configureAndBuild(repository, buildDirectory, sourceDirectory);
            Path library = buildDirectory.resolve("lib").resolve(
                System.mapLibraryName("jlc_native"));
            assertTrue("custom generated JNI library was not built", Files.isRegularFile(library));
            System.setProperty("jlc.native.lib.path", library.toString());

            double[][] inputs = {a.getRawData(), b.getRawData()};
            double[] output = new double[a.getRawData().length];
            String signature = avx2.signature().canonicalText();
            for (int warmup = 0; warmup < 10; warmup++) {
                assertTrue(NativeGeneratedKernelSupport.execute(
                    signature, inputs, output, 256, 256));
            }
            long[] samples = new long[21];
            for (int index = 0; index < samples.length; index++) {
                long start = System.nanoTime();
                assertTrue(NativeGeneratedKernelSupport.execute(
                    signature, inputs, output, 256, 256));
                samples[index] = System.nanoTime() - start;
            }
            Arrays.sort(samples);

            double[][] emptyInputs = {new double[0], new double[0]};
            double[] emptyOutput = new double[0];
            String zeroSignature = zeroScalar.signature().canonicalText();
            for (int warmup = 0; warmup < 50; warmup++) {
                assertTrue(NativeGeneratedKernelSupport.execute(
                    zeroSignature, emptyInputs, emptyOutput, 0, 3));
            }
            long[] boundarySamples = new long[101];
            for (int index = 0; index < boundarySamples.length; index++) {
                long start = System.nanoTime();
                assertTrue(NativeGeneratedKernelSupport.execute(
                    zeroSignature, emptyInputs, emptyOutput, 0, 3));
                boundarySamples[index] = System.nanoTime() - start;
            }
            Arrays.sort(boundarySamples);

            double elements = 256.0 * 256.0;
            System.out.printf("r4_jni_shape=256x256 median_ns=%.1f ns_per_element=%.4f%n",
                (double) samples[samples.length / 2],
                samples[samples.length / 2] / elements);
            System.out.printf("r4_jni_empty_shape=0x3 boundary_median_ns=%.1f%n",
                (double) boundarySamples[boundarySamples.length / 2]);
        } finally {
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

    private static void configureAndBuild(Path repository,
                                          Path buildDirectory,
                                          Path sourceDirectory) throws Exception {
        List<String> configure = List.of(
            "cmake", "-S", repository.resolve("native-backend").toString(),
            "-B", buildDirectory.toString(),
            "-DJLC_JAVA_HOME=" + Path.of(System.getProperty("java.home")).toAbsolutePath(),
            "-DJLC_NATIVE_ENABLE_VENDOR_BLAS=OFF",
            "-DJLC_NATIVE_ENABLE_MARCH_NATIVE=OFF",
            "-DJLC_R4_GENERATED_SOURCE_DIR=" + sourceDirectory);
        run(configure, repository, 60);
        run(List.of("cmake", "--build", buildDirectory.toString(),
            "--target", "jlc_native", "-j2"), repository, 120);
    }

    private static void run(List<String> command, Path directory, long timeoutSeconds)
        throws Exception {
        Process process = new ProcessBuilder(new ArrayList<>(command))
            .directory(directory.toFile()).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue("command failed: " + command + "\n" + output,
            process.waitFor(timeoutSeconds, TimeUnit.SECONDS) && process.exitValue() == 0);
    }

    private static KernelLoweringResult lower(MatrixExpr expression) {
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        return KernelLowerer.lower(program.cpuPlan().fusedRegions().get(0));
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + index * 0.0001;
        }
        return Matrix.wrap(data, rows, columns);
    }

    private static boolean commandWorks(String command, String argument) {
        try {
            Process process = new ProcessBuilder(command, argument)
                .redirectErrorStream(true).start();
            return process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException | InterruptedException failure) {
            if (failure instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }
}
