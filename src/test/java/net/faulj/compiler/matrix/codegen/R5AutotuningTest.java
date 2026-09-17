package net.faulj.compiler.matrix.codegen;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
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
import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelLowerer;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;
import net.faulj.compiler.matrix.kernel.KernelReferenceExecutor;
import net.faulj.matrix.Matrix;

import org.junit.After;
import org.junit.Before;
import org.junit.Assume;
import org.junit.Test;

/** Focused R5 proof for deterministic search, gating, selection, and profile safety. */
public class R5AutotuningTest {
    private final GeneratedKernelRegistry registry = GeneratedKernelRegistry.global();

    @Before
    public void setUp() {
        registry.clear();
        KernelDispatchSelector.resetForTests();
    }

    @After
    public void tearDown() {
        registry.clear();
        KernelDispatchSelector.resetForTests();
    }

    @Test
    public void enumerationAndVariantSourceIdentityAreDeterministic() {
        PseudokernelPlan plan = lower(3, 17);
        List<KernelVariantCandidate> first = KernelVariantGenerator.enumerate(plan);
        List<KernelVariantCandidate> second = KernelVariantGenerator.enumerate(plan);
        assertEquals(first.toString(), second.toString());
        assertTrue(first.size() <= KernelVariantGenerator.DEFAULT_MAX_CANDIDATES);
        assertTrue(first.stream().anyMatch(item -> item.signature().isBaselineAvx2()));

        for (KernelVariantCandidate candidate : first) {
            assertEquals(candidate.signature(),
                new KernelVariantSignature(plan.signature(), candidate.signature().backend(),
                    candidate.signature().isa(), candidate.signature().vectorWidth(),
                    candidate.signature().unroll(), candidate.signature().loopForm(),
                    candidate.signature().tailPolicy(), candidate.signature().semantics(),
                    candidate.signature().fmaContraction()));
            if (candidate.eligible()) {
                GeneratedKernelSource a = KernelCodeGenerator.variant(
                    plan, candidate.signature(), CppEmissionOptions.nativeRegistry());
                GeneratedKernelSource b = KernelCodeGenerator.variant(
                    plan, candidate.signature(), CppEmissionOptions.nativeRegistry());
                assertEquals(a.source(), b.source());
                assertEquals(candidate.signature().generatedSymbol(), a.symbol());
                if (candidate.signature().semantics() == OptimizationSemantics.STRICT) {
                    assertFalse(a.source().contains("fmadd"));
                }
            }
        }
    }

    @Test
    public void tunerCorrectnessGatesCandidatesAndPromotesClearStableWinner() throws Exception {
        PseudokernelPlan plan = lower(3, 17);
        List<KernelBinding> corpus = KernelValidationCorpus.forPlan(plan);
        KernelVariantCompiler compiler = (ignoredPlan, candidate) ->
            CompiledKernelVariant.of(candidate, referenceInvoker(plan));
        KernelTuningConfig config = new KernelTuningConfig(0, 9, 10, 0L, 0.10, 0.03);
        KernelTuningResult result = KernelVariantTuner.tune(
            plan, compiler, corpus, corpus.get(0), config,
            (compiled, ignoredBinding, ignoredConfig) -> {
                long median = compiled.variantSignature().variantId().equals("avx2_u4_flat")
                    ? 80L : compiled.variantSignature().isBaselineAvx2() ? 100L : 110L;
                return KernelBenchmarkStatistics.from(new long[] {
                    median - 2, median - 1, median, median, median, median + 1,
                    median + 1, median + 2, median
                });
            });

        assertEquals(KernelVariantSignature.baselineAvx2(plan.signature()), result.baseline());
        assertEquals("avx2_u4_flat", result.winner().variantId());
        assertTrue(result.winnerSpeedup() >= 1.20);
        assertTrue(result.candidates().stream().allMatch(item ->
            item.outcome() != KernelCandidateOutcome.PASS || item.correctnessPassed()));
        assertTrue(result.candidates().stream().anyMatch(item ->
            item.outcome() == KernelCandidateOutcome.PASS
                && !item.variantSignature().equals(result.winner())));
    }

    @Test
    public void profileRoundTripSelectsOnlyMatchingStableRegisteredWinner() throws Exception {
        PseudokernelPlan plan = lower(3, 17);
        List<KernelBinding> corpus = KernelValidationCorpus.forPlan(plan);
        KernelVariantCompiler compiler = (ignoredPlan, candidate) ->
            CompiledKernelVariant.of(candidate, referenceInvoker(plan));
        KernelTuningResult result = KernelVariantTuner.tune(
            plan, compiler, corpus, corpus.get(0),
            new KernelTuningConfig(0, 9, 10, 0L, 0.10, 0.03),
            (compiled, ignoredBinding, ignoredConfig) -> KernelBenchmarkStatistics.from(
                new long[] {100, 100, 100, 100, 100, 100, 100, 100, 100}));
        KernelMachineIdentity machine = new KernelMachineIdentity(
            "x86_64", "GenuineIntel", "6", "85", "test-cpu", "avx,avx2,fma",
            "Linux/test", "21/test", "c++ test", 8);
        KernelBuildIdentity build = new KernelBuildIdentity(
            "test", "abc123", "kernel-v1", "variant-v1", "c++ test");
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromResult(machine, build, result);
        assertEquals(3, result.trustedBaselines().size());
        Path path = Files.createTempFile("jlc-r5-profile-", ".json");
        try {
            KernelCalibrationProfileStore.save(path, profile);
            KernelCalibrationProfile loaded = KernelCalibrationProfileStore.load(path).orElseThrow();
            assertEquals("matched", loaded.compatibilityReason(machine, build, plan.signature()));
            KernelBuildIdentity staleBuild = new KernelBuildIdentity(
                "test", "abc123", "kernel-v1", "variant-v1", "c++ test",
                "old-codegen-abi", KernelBuildIdentity.STRICT_FLAGS);
            KernelCalibrationProfile stale = new KernelCalibrationProfile(
                machine, staleBuild, loaded.timestamp(), loaded.methodologyVersion(),
                loaded.entries());
            assertEquals("build mismatch",
                stale.compatibilityReason(machine, build, plan.signature()));
            assertEquals("schema version mismatch",
                KernelCalibrationProfile.fromJson(
                    loaded.toJson().replace("\"schemaVersion\" : 1",
                        "\"schemaVersion\" : 99"))
                    .compatibilityReason(machine, build, plan.signature()));
            for (KernelVariantCandidate candidate : KernelVariantGenerator.enumerate(plan)) {
                if (!candidate.eligible()) continue;
                GeneratedKernelSource source = KernelCodeGenerator.variant(plan, candidate.signature());
                registry.register(source.descriptor(), referenceInvoker(plan));
            }
            KernelDispatchSelector selector = new KernelDispatchSelector(
                registry, machine, build, path);
            KernelDispatchSelector.Selection selected = selector.select(plan.signature());
            assertNotNull(selected.entry());
            assertTrue(selected.explain().contains("profile"));
            assertTrue(selector.resolvedCacheSize() == 1);
            long lookupNanos = selector.measureLookupNanos(plan.signature(), 10_000);
            System.out.println("r5_selector_lookup_nanos=" + lookupNanos);
            assertTrue(lookupNanos > 0L);

            KernelMachineIdentity mismatch = new KernelMachineIdentity(
                "x86_64", "GenuineIntel", "6", "86", "other-cpu", "avx,avx2,fma",
                "Linux/test", "21/test", "c++ test", 8);
            KernelDispatchSelector cold = new KernelDispatchSelector(registry, mismatch, build, path);
            assertFalse(cold.select(plan.signature()).profileHit());
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void strictVariantCannotRequestFmaAndCorruptProfileIsIgnored() throws Exception {
        PseudokernelPlan plan = lower(2, 5);
        try {
            new KernelVariantSignature(plan.signature(), CodegenBackend.AVX2, "avx2", 4, 1,
                KernelLoopForm.FLAT, KernelTailPolicy.SCALAR, OptimizationSemantics.STRICT,
                FmaContraction.EXPLICIT);
            throw new AssertionError("STRICT FMA variant was accepted");
        } catch (IllegalArgumentException expected) {
            // required safety gate
        }
        Path corrupt = Files.createTempFile("jlc-r5-corrupt-", ".json");
        try {
            Files.writeString(corrupt, "{not-json");
            assertTrue(KernelCalibrationProfileStore.load(corrupt).isEmpty());
        } finally {
            Files.deleteIfExists(corrupt);
        }
    }

    @Test
    public void allEligibleVariantsCoexistAndExecuteInOneNativeRegistryWhenRequested() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("jlc.compiler.r5.native"));
        Assume.assumeTrue(commandWorks("c++", "--version"));
        PseudokernelPlan plan = lower(3, 17);
        List<GeneratedKernelSource> sources = KernelVariantGenerator.enumerate(plan).stream()
            .filter(KernelVariantCandidate::eligible)
            .map(candidate -> KernelCodeGenerator.variant(
                plan, candidate.signature(), CppEmissionOptions.nativeRegistry()))
            .toList();
        Path repository = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        Path directory = Files.createTempDirectory("jlc-r5-native-");
        try {
            for (GeneratedKernelSource source : sources) {
                Files.writeString(directory.resolve(source.symbol() + ".cpp"),
                    source.source(), StandardCharsets.UTF_8);
            }
            Files.writeString(directory.resolve("driver.cpp"), nativeDriver(sources),
                StandardCharsets.UTF_8);
            Path header = repository.resolve(
                "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.h");
            Path registrySource = repository.resolve(
                "native-backend/src/main/cpp/codegen/jlc_generated_kernel_registry.cpp");
            Path executable = directory.resolve("r5_variants");
            List<String> command = new ArrayList<>(List.of(
                "c++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                "-mavx2", "-I", header.getParent().toString()));
            sources.forEach(source -> command.add(directory.resolve(source.symbol() + ".cpp").toString()));
            command.add(directory.resolve("driver.cpp").toString());
            command.add(registrySource.toString());
            command.add("-o");
            command.add(executable.toString());
            run(command, repository, 60);
            run(List.of(executable.toString()), repository, 30);
        } finally {
            try (Stream<Path> paths = Files.walk(directory)) {
                paths.sorted((left, right) -> right.getNameCount() - left.getNameCount())
                    .forEach(path -> {
                        try { Files.deleteIfExists(path); } catch (Exception ignored) { }
                    });
            }
        }
    }

    private static GeneratedKernelInvoker referenceInvoker(PseudokernelPlan plan) {
        return binding -> {
            KernelReferenceExecutor.executeVerified(plan.function(), binding);
            return true;
        };
    }

    private static PseudokernelPlan lower(int rows, int columns) {
        Matrix a = matrix(rows, columns, 1.0);
        Matrix b = matrix(rows, columns, 2.0);
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)),
            OptimizationSemantics.STRICT, new FlopCostModel(), FusionStrategy.GENERALIZED);
        KernelLoweringResult lowering = KernelLowerer.lower(
            program.cpuPlan().fusedRegions().get(0));
        return PseudokernelPlanner.plan(lowering.program().function());
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) data[index] = start + index * 0.01;
        return Matrix.wrap(data, rows, columns);
    }

    private static String nativeDriver(List<GeneratedKernelSource> sources) {
        String signature = CppEmitterSupport.escape(sources.get(0).signature().canonicalText());
        StringBuilder driver = new StringBuilder()
            .append("#include <cstddef>\n#include <vector>\n#include \"jlc_generated_kernel_registry.h\"\n")
            .append("int main() { const std::size_t rows=3, cols=17, count=rows*cols;\n")
            .append("double a[count], b[count], out[count];\n")
            .append("for (std::size_t p=0; p<count; ++p) { a[p]=1.0; b[p]=2.0; }\n")
            .append("const double* inputs[] = {a,b};\n");
        for (GeneratedKernelSource source : sources) {
            driver.append("if (!jlc_generated_execute_variant(\"")
                .append(signature).append("\",\"")
                .append(CppEmitterSupport.escape(source.variantSignature().canonicalText()))
                .append("\",inputs,2,out,rows,cols)) return 2;\n")
                .append("for (std::size_t p=0; p<count; ++p) if (out[p] != 4.0) return 3;\n");
        }
        return driver.append("return 0; }\n").toString();
    }

    private static void run(List<String> command, Path directory, long timeoutSeconds)
        throws Exception {
        Process process = new ProcessBuilder(command).directory(directory.toFile())
            .redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        boolean completed = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
        assertTrue("command failed: " + command + " exit="
                + (completed ? Integer.toString(process.exitValue()) : "timeout") + "\n" + output,
            completed && process.exitValue() == 0);
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
}
