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

import org.junit.Assume;
import org.junit.Test;

/**
 * Opt-in R5 direct-native benchmark. Compilation is outside timed regions;
 * every exact variant is correctness-checked before the timed loop.
 */
public class R5GeneratedKernelBenchmarkTest {
    @Test
    public void benchmarkBoundedVariantSpaceWhenRequested() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jlc.compiler.r5.benchmark"));
        Assume.assumeTrue(commandWorks("c++", "--version"));
        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        List<Integer> sizes = parseList("jlc.compiler.r5.benchmark.sizes", "64,256");
        List<Integer> operations = parseList("jlc.compiler.r5.benchmark.operations", "2,10,25");
        Path report = repository.resolve("build/reports/r5/variant-benchmark.txt");
        Files.createDirectories(report.getParent());
        StringBuilder output = new StringBuilder(
            "# R5 direct-native variant benchmark\n"
                + "# samples=11 warmup=10 statistics=median/min/max/MAD\n");
        for (int size : sizes) {
            for (int arithmeticOps : operations) {
                Workload workload = workload(size, arithmeticOps);
                RunResult result = runNative(repository, workload);
                output.append(result.text());
                System.out.print(result.text());
            }
        }
        Files.writeString(report, output.toString(), StandardCharsets.UTF_8);
    }

    private static RunResult runNative(Path repository, Workload workload) throws Exception {
        Path header = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.h");
        Path registry = repository.resolve(
            "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.cpp");
        Path directory = Files.createTempDirectory("jlc-r5-benchmark-");
        long generationStart = System.nanoTime();
        try {
            List<GeneratedKernelSource> sources = workload.candidates().stream()
                .filter(KernelVariantCandidate::eligible)
                .map(candidate -> KernelCodeGenerator.variant(workload.plan(), candidate.signature(),
                    CppEmissionOptions.nativeRegistry()))
                .toList();
            long generationNanos = System.nanoTime() - generationStart;
            for (GeneratedKernelSource source : sources) {
                Files.writeString(directory.resolve(source.symbol() + ".cpp"), source.source(),
                    StandardCharsets.UTF_8);
            }
            Files.writeString(directory.resolve("driver.cpp"), driver(workload, sources),
                StandardCharsets.UTF_8);
            Path executable = directory.resolve("r5_benchmark");
            List<String> compile = new ArrayList<>(List.of(
                "c++", "-std=c++17", "-O3", "-fno-fast-math", "-ffp-contract=off",
                "-mavx2", "-I", header.getParent().toString()));
            sources.forEach(source -> compile.add(directory.resolve(source.symbol() + ".cpp").toString()));
            compile.add(directory.resolve("driver.cpp").toString());
            compile.add(registry.toString());
            compile.add("-o");
            compile.add(executable.toString());
            long compileStart = System.nanoTime();
            run(compile, repository, 120);
            long compilationNanos = System.nanoTime() - compileStart;
            Process process = new ProcessBuilder(executable.toString()).directory(repository.toFile())
                .redirectErrorStream(true).start();
            String text = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue("benchmark failed:\n" + text,
                process.waitFor(120, TimeUnit.SECONDS) && process.exitValue() == 0);
            KernelWorkloadAnalysis analysis = KernelWorkloadAnalysis.from(workload.plan());
            String pressure = workload.candidates().stream()
                .map(candidate -> candidate.signature().variantId() + "="
                    + candidate.estimatedLiveVectorRegisters()
                    + (candidate.isPruned() ? "(pruned:" + candidate.reason() + ")" : ""))
                .reduce((left, right) -> left + "," + right).orElse("");
            return new RunResult("workload=" + workload.name() + " shape=" + workload.size()
                + " generation_us=" + generationNanos / 1_000L
                + " compilation_us=" + compilationNanos / 1_000L
                + " fp_ops_per_element=" + analysis.estimatedFpOpsPerElement()
                + " bytes_per_element=" + analysis.estimatedBytesPerElement()
                + " arithmetic_intensity=" + analysis.arithmeticIntensity()
                + " class=" + analysis.classification()
                + " max_live_vectors=" + workload.plan().maxLiveVectorValues()
                + " pressure={" + pressure + "}\n" + text);
        } finally {
            try (Stream<Path> paths = Files.walk(directory)) {
                paths.sorted((left, right) -> right.getNameCount() - left.getNameCount())
                    .forEach(path -> {
                        try { Files.deleteIfExists(path); } catch (IOException ignored) { }
                    });
            }
        }
    }

    private static String driver(Workload workload, List<GeneratedKernelSource> sources) {
        String signature = CppEmitterSupport.escape(workload.plan().signature().canonicalText());
        int inputCount = workload.plan().function().inputBuffers().size();
        int size = workload.size();
        StringBuilder driver = new StringBuilder()
            .append("#include <algorithm>\n#include <chrono>\n#include <cmath>\n")
            .append("#include <cstddef>\n#include <cstdio>\n#include <vector>\n")
            .append("#include \"jlc_generated_kernel_registry.h\"\n")
            .append("volatile double sink = 0.0;\nint main() {\n")
            .append("const std::size_t rows=").append(size).append(", cols=").append(size)
            .append(", count=rows*cols;\n")
            .append("std::vector<std::vector<double>> storage(").append(inputCount)
            .append(", std::vector<double>(count));\n")
            .append("for (std::size_t input=0; input<").append(inputCount)
            .append("; ++input) for (std::size_t p=0; p<count; ++p) storage[input][p] = ")
            .append("0.125 + input * 0.375 + (p % 97) * 0.001;\n")
            .append("std::vector<const double*> inputs; for (auto& values: storage) inputs.push_back(values.data());\n")
            .append("std::vector<double> output(count), expected(count);\n");
        driver.append("for (std::size_t p=0; p<count; ++p) { ")
            .append("double value=storage[0][p];\n");
        for (int operation = 0; operation < workload.arithmeticOps(); operation++) {
            if ((operation & 1) == 0) {
                driver.append("value *= ").append(1.001 + operation * 0.0001).append(";\n");
            } else {
                driver.append("value += storage[1][p];\n");
            }
        }
        driver.append("expected[p]=value; }\n");
        for (GeneratedKernelSource source : sources) {
            driver.append("{\nfor (int warmup=0; warmup<10; ++warmup) if (!jlc_generated_execute_variant(\"")
                .append(signature).append("\",\"")
                .append(CppEmitterSupport.escape(source.variantSignature().canonicalText()))
                .append("\", inputs.data(), ").append(inputCount)
                .append(", output.data(), rows, cols)) return 2;\n")
                .append("for (std::size_t p=0; p<count; ++p) if (output[p] != expected[p]) return 3;\n")
                .append("long long samples[11]; for (int sample=0; sample<11; ++sample) { auto start=std::chrono::steady_clock::now();\n")
                .append("if (!jlc_generated_execute_variant(\"").append(signature).append("\",\"")
                .append(CppEmitterSupport.escape(source.variantSignature().canonicalText()))
                .append("\", inputs.data(), ").append(inputCount)
                .append(", output.data(), rows, cols)) return 4; auto stop=std::chrono::steady_clock::now();\n")
                .append("samples[sample]=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count(); sink += output[sample % count]; }\n")
                .append("std::sort(samples,samples+11); double median=samples[5]; double deviations[11];\n")
                .append("for (int i=0;i<11;++i) deviations[i]=std::abs((double)samples[i]-median); std::sort(deviations,deviations+11);\n")
                .append("std::printf(\"variant=").append(source.variantSignature().variantId())
                .append(" median_ns=%.1f min_ns=%lld max_ns=%lld mad_ns=%.1f\\n\", median, samples[0], samples[10], deviations[5]);\n}\n");
        }
        return driver.append("return 0; }\n").toString();
    }

    private static Workload workload(int size, int arithmeticOps) {
        Matrix a = matrix(size, size, 1.0);
        Matrix b = matrix(size, size, 2.0);
        MatrixExpr expression = MatrixExpr.input(a);
        for (int operation = 0; operation < arithmeticOps; operation++) {
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
        return new Workload(arithmeticOps + "-op", size, arithmeticOps, plan,
            KernelVariantGenerator.enumerate(plan));
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        Arrays.fill(data, start);
        return Matrix.wrap(data, rows, columns);
    }

    private static List<Integer> parseList(String property, String fallback) {
        return Arrays.stream(System.getProperty(property, fallback).split(","))
            .map(String::trim).filter(value -> !value.isEmpty()).map(Integer::parseInt).toList();
    }

    private static void run(List<String> command, Path directory, long timeoutSeconds)
        throws Exception {
        Process process = new ProcessBuilder(command).directory(directory.toFile())
            .redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue("command failed: " + command + "\n" + output,
            process.waitFor(timeoutSeconds, TimeUnit.SECONDS) && process.exitValue() == 0);
    }

    private static boolean commandWorks(String command, String argument) {
        try {
            Process process = new ProcessBuilder(command, argument).redirectErrorStream(true).start();
            return process.waitFor(10, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException | InterruptedException failure) {
            if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
            return false;
        }
    }

    private record Workload(String name, int size, int arithmeticOps,
                            PseudokernelPlan plan, List<KernelVariantCandidate> candidates) {
    }

    private record RunResult(String text) {
    }
}
