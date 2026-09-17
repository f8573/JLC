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
import net.faulj.matrix.Matrix;

import org.junit.Assume;
import org.junit.Test;

/**
 * Opt-in R4 execution benchmark. Native compilation is outside timed regions;
 * the driver prints median execution nanoseconds only.
 */
public class R4GeneratedKernelBenchmarkTest {
    @Test
    public void benchmarkGeneratedKernelsWhenRequested() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jlc.compiler.r4.benchmark"));
        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path registryHeader = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.h");
        Path registrySource = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.cpp");
        Assume.assumeTrue(Files.isRegularFile(registryHeader));
        Assume.assumeTrue(Files.isRegularFile(registrySource));
        Assume.assumeTrue(commandWorks("c++", "--version"));

        List<Integer> sizes = parseList("jlc.compiler.r4.benchmark.sizes", "64,128,256,512,1024");
        List<Integer> complexities = parseList(
            "jlc.compiler.r4.benchmark.complexities", "2,5,10,25");
        System.out.println("| Kernel | Shape | R2 Java | Scalar C++ | AVX2 | AVX2 vs R2 | Correct |");
        System.out.println("|---|---:|---:|---:|---:|---:|---|");
        for (int size : sizes) {
            for (int complexity : complexities) {
                Matrix a = matrix(size, size, 1.0);
                Matrix b = matrix(size, size, 2.0);
                MatrixExpr expression = expression(a, b, complexity);
                CompiledMatrixProgram program = MatrixCompiler.compileProgram(
                    expression, OptimizationSemantics.STRICT, new FlopCostModel(),
                    FusionStrategy.GENERALIZED);
                PseudokernelPlan plan = PseudokernelPlanner.plan(
                    KernelLowerer.lower(program.cpuPlan().fusedRegions().get(0))
                        .program().function());
                long r2 = medianJava(program, size);
                long scalar = runNative(repository, registryHeader, registrySource,
                    KernelCodeGenerator.scalar(plan, CppEmissionOptions.nativeRegistry()), false);
                long avx2 = RuntimeCpuFeatures.avx2Supported()
                    ? runNative(repository, registryHeader, registrySource,
                        KernelCodeGenerator.avx2(plan, CppEmissionOptions.nativeRegistry()), true)
                    : -1L;
                String avxRatio = avx2 < 0L ? "n/a" : String.format("%.2fx", (double) r2 / avx2);
                System.out.printf("| %d ops | %dx%d | %.1f | %.1f | %s | %s | yes |%n",
                    complexity, size, size, r2 / (double) (size * (long) size),
                    scalar / (double) (size * (long) size),
                    avx2 < 0L ? "n/a" : String.format("%.1f", avx2 / (double) (size * (long) size)),
                    avxRatio);
            }
        }
    }

    private static long medianJava(CompiledMatrixProgram program, int size) {
        for (int warmup = 0; warmup < 5; warmup++) {
            program.execute();
        }
        long[] samples = new long[15];
        for (int sample = 0; sample < samples.length; sample++) {
            long start = System.nanoTime();
            program.execute();
            samples[sample] = System.nanoTime() - start;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2];
    }

    private static long runNative(Path repository,
                                  Path registryHeader,
                                  Path registrySource,
                                  GeneratedKernelSource generated,
                                  boolean avx2) throws Exception {
        Path directory = Files.createTempDirectory("jlc-r4-bench-");
        long generationAndWriteStart = System.nanoTime();
        try {
            Path source = directory.resolve(generated.symbol() + ".cpp");
            Path driver = directory.resolve("driver.cpp");
            Path executable = directory.resolve("r4_bench");
            Files.writeString(source, generated.source(), StandardCharsets.UTF_8);
            Files.writeString(driver, benchmarkDriver(generated), StandardCharsets.UTF_8);
            List<String> command = new ArrayList<>(List.of(
                "c++", "-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off",
                "-I", registryHeader.getParent().toString(), source.toString(), driver.toString(),
                registrySource.toString(), "-o", executable.toString()));
            if (avx2) {
                command.add(1, "-mavx2");
            }
            Process compile = new ProcessBuilder(command).directory(repository.toFile())
                .redirectErrorStream(true).start();
            String compileOutput = new String(compile.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue("benchmark generated source did not compile:\n" + compileOutput,
                compile.waitFor(60, TimeUnit.SECONDS) && compile.exitValue() == 0);
            Process execute = new ProcessBuilder(executable.toString()).directory(repository.toFile())
                .redirectErrorStream(true).start();
            String output = new String(execute.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue("benchmark driver failed: " + output,
                execute.waitFor(60, TimeUnit.SECONDS) && execute.exitValue() == 0);
            for (String line : output.lines().toList()) {
                if (line.startsWith("median_ns=")) {
                    System.out.println("r4_codegen_time_us="
                        + generated.generationTimeMicros()
                        + " compile_wall_us="
                        + (System.nanoTime() - generationAndWriteStart) / 1_000L);
                    return Long.parseLong(line.substring("median_ns=".length()).trim());
                }
            }
            throw new AssertionError("benchmark driver did not report median: " + output);
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

    private static String benchmarkDriver(GeneratedKernelSource generated) {
        int inputCount = generated.plan().function().inputBuffers().size();
        String signature = CppEmitterSupport.escape(generated.signature().canonicalText());
        StringBuilder source = new StringBuilder()
            .append("#include <cstddef>\n#include <cstdio>\n#include <vector>\n")
            .append("#include \"jlc_generated_kernel_registry.h\"\n")
            .append("int main() {\n")
            .append("  const std::size_t rows = ").append(generated.plan().function().loops().get(0).upperBound())
            .append(", cols = ").append(generated.plan().function().loops().get(1).upperBound()).append(";\n")
            .append("  const std::size_t count = rows * cols;\n")
            .append("  std::vector<std::vector<double>> storage(").append(inputCount)
            .append(", std::vector<double>(count));\n")
            .append("  std::vector<const double*> inputs; inputs.reserve(").append(inputCount).append(");\n")
            .append("  for (std::size_t input = 0; input < ").append(inputCount).append("; ++input) {\n")
            .append("    for (std::size_t p = 0; p < count; ++p) storage[input][p] = 0.25 + input + p * 0.0001;\n")
            .append("    inputs.push_back(storage[input].data());\n  }\n")
            .append("  std::vector<double> output(count);\n")
            .append("  for (int warmup = 0; warmup < 10; ++warmup)\n")
            .append("    if (!jlc_generated_execute(\"").append(signature)
            .append("\", inputs.data(), inputs.size(), output.data(), rows, cols)) return 2;\n")
            .append("  long long samples[21];\n")
            .append("  for (int sample = 0; sample < 21; ++sample) {\n")
            .append("    auto start = std::chrono::steady_clock::now();\n")
            .append("    if (!jlc_generated_execute(\"").append(signature)
            .append("\", inputs.data(), inputs.size(), output.data(), rows, cols)) return 3;\n")
            .append("    auto stop = std::chrono::steady_clock::now();\n")
            .append("    samples[sample] = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();\n  }\n")
            .append("  std::sort(samples, samples + 21); std::printf(\"median_ns=%lld\\n\", samples[10]); return 0;\n}\n");
        return "#include <algorithm>\n#include <chrono>\n" + source;
    }

    private static MatrixExpr expression(Matrix a, Matrix b, int operations) {
        MatrixExpr result = MatrixExpr.input(a);
        for (int operation = 0; operation < operations; operation++) {
            if ((operation & 1) == 0) {
                result = result.scale(1.001 + operation * 0.0001);
            } else {
                result = result.add(MatrixExpr.input(b));
            }
        }
        return result;
    }

    private static List<Integer> parseList(String property, String fallback) {
        String configured = System.getProperty(property, fallback);
        return Arrays.stream(configured.split(","))
            .map(String::trim).filter(value -> !value.isEmpty()).map(Integer::parseInt).toList();
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
            Process process = new ProcessBuilder(command, argument).redirectErrorStream(true).start();
            return process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException | InterruptedException failure) {
            if (failure instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return false;
        }
    }
}
