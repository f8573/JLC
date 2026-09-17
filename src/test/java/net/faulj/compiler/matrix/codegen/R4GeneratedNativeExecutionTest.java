package net.faulj.compiler.matrix.codegen;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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

import org.junit.Assume;
import org.junit.Test;

/** Compile-smoke and native whole-region execution evidence for R4. */
public class R4GeneratedNativeExecutionTest {
    @Test
    public void scalarAndAvx2SourcesCompileAndExecuteThroughRegistry() throws Exception {
        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path registryHeader = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.h");
        Path registrySource = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.cpp");
        Assume.assumeTrue(Files.isRegularFile(registryHeader));
        Assume.assumeTrue(Files.isRegularFile(registrySource));
        Assume.assumeTrue(commandWorks("c++", "--version"));

        PseudokernelPlan plan = lower(MatrixExpr.input(matrix(3, 5, 1.0)).scale(2.0)
            .add(MatrixExpr.input(matrix(3, 5, 4.0))));
        GeneratedKernelSource scalar = KernelCodeGenerator.scalar(
            plan, CppEmissionOptions.nativeRegistry());
        runGenerated(repository, registryHeader, registrySource, scalar, false);

        Assume.assumeTrue(RuntimeCpuFeatures.avx2Supported());
        GeneratedKernelSource avx2 = KernelCodeGenerator.avx2(
            plan, CppEmissionOptions.nativeRegistry());
        runGenerated(repository, registryHeader, registrySource, avx2, true);
    }

    private static void runGenerated(Path repository,
                                     Path registryHeader,
                                     Path registrySource,
                                     GeneratedKernelSource generated,
                                     boolean avx2) throws Exception {
        Path directory = Files.createTempDirectory("jlc-r4-generated-");
        try {
            Path generatedSource = directory.resolve(generated.symbol() + ".cpp");
            Path driver = directory.resolve("driver.cpp");
            Path executable = directory.resolve("r4_generated");
            Files.writeString(generatedSource, generated.source(), StandardCharsets.UTF_8);
            Files.writeString(driver, driverSource(generated), StandardCharsets.UTF_8);
            List<String> command = new ArrayList<>(List.of(
                "c++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                "-I", registryHeader.getParent().toString(),
                generatedSource.toString(), driver.toString(), registrySource.toString(),
                "-o", executable.toString()));
            if (Boolean.getBoolean("jlc.compiler.r4.asan")) {
                command.add(1, "-fno-omit-frame-pointer");
                command.add(1, "-fsanitize=address");
            }
            if (avx2) {
                command.add(1, "-mavx2");
            }
            Process compile = new ProcessBuilder(command)
                .directory(repository.toFile())
                .redirectErrorStream(true)
                .start();
            String compileOutput = new String(compile.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue("generated source did not compile:\n" + compileOutput,
                compile.waitFor(30, TimeUnit.SECONDS) && compile.exitValue() == 0);

            Process execute = new ProcessBuilder(executable.toString())
                .directory(repository.toFile())
                .redirectErrorStream(true)
                .start();
            String output = new String(execute.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue("generated executable failed: " + output,
                execute.waitFor(30, TimeUnit.SECONDS) && execute.exitValue() == 0);
        } finally {
            try (Stream<Path> paths = Files.walk(directory)) {
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

    private static String driverSource(GeneratedKernelSource generated) {
        String signature = CppEmitterSupport.escape(generated.signature().canonicalText());
        String symbol = generated.symbol();
        return "#include <cmath>\n"
            + "#include <cstddef>\n"
            + "#include <limits>\n"
            + "#include \"jlc_generated_kernel_registry.h\"\n"
            + "int main() {\n"
            + "  constexpr std::size_t rows = 3, cols = 5;\n"
            + "  double a[rows * cols]; double b[rows * cols]; double out[rows * cols];\n"
            + "  for (std::size_t p = 0; p < rows * cols; ++p) { a[p] = 1.0 + p; b[p] = 4.0 + p; out[p] = -99.0; }\n"
            + "  a[0] = std::numeric_limits<double>::quiet_NaN();\n"
            + "  const double* inputs[] = {a, b};\n"
            + "  if (!jlc_generated_execute(\"" + signature + "\", inputs, 2, out, rows, cols)) return 2;\n"
            + "  if (!std::isnan(out[0])) return 3;\n"
            + "  for (std::size_t p = 1; p < rows * cols; ++p) if (out[p] != 6.0 + 3.0 * p) return 4;\n"
            + "  return 0;\n"
            + "}\n";
    }

    private static boolean commandWorks(String command, String argument) {
        try {
            Process process = new ProcessBuilder(command, argument)
                .redirectErrorStream(true).start();
            return process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException | InterruptedException failure) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static PseudokernelPlan lower(MatrixExpr expression) {
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        KernelLoweringResult lowering = KernelLowerer.lower(
            program.cpuPlan().fusedRegions().get(0));
        return PseudokernelPlanner.plan(lowering.program().function());
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + index;
        }
        return Matrix.wrap(data, rows, columns);
    }
}
