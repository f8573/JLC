package net.faulj.compiler.matrix.codegen;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.kernel.KernelLowerer;
import net.faulj.compiler.matrix.kernel.KernelReferenceExecutor;
import net.faulj.matrix.Matrix;

import org.junit.Test;

/** Focused proof of the typed post-R5 dispatch boundary. */
public class BackendDispatchIntegrationTest {
    @Test
    public void allTypedChoicesRoundTripAndUnknownTagsAreRejected() {
        PseudokernelPlan plan = lower(2, 5);
        KernelVariantSignature scalarVariant = KernelVariantSignature.legacyScalar(plan.signature());
        KernelVariantSignature avxVariant = KernelVariantSignature.baselineAvx2(plan.signature());
        List<BackendCandidateEvidence> evidence = List.of(
            pass(new R2JavaBackend(), 100),
            pass(new ScalarNativeBackend(scalarVariant), 90),
            pass(new GeneratedAvx2Backend(avxVariant), 80));
        BackendCalibrationResult result = new BackendCalibrationResult(
            plan.signature(), "2x2/ops=5", new GeneratedAvx2Backend(avxVariant), evidence,
            new R2JavaBackend(), 0.8, "Java wins for this small shape");
        KernelMachineIdentity machine = testMachine();
        KernelBuildIdentity build = testBuild();
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            machine, build, result);
        KernelCalibrationProfile loaded = KernelCalibrationProfile.fromJson(profile.toJson());
        KernelCalibrationEntry entry = loaded.entry(plan.signature()).orElseThrow();
        assertTrue(entry.baselineChoice() instanceof GeneratedAvx2Backend);
        assertTrue(entry.winnerChoice() instanceof R2JavaBackend);
        assertEquals(3, entry.backendCandidates().size());
        assertEquals("matched", loaded.compatibilityReason(machine, build, plan.signature()));

        KernelCalibrationProfile oldR5 = KernelCalibrationProfile.fromJson(
            profile.toJson().replace("  \"dispatchSchemaVersion\" : 3,\n", ""));
        assertEquals("dispatch schema version mismatch",
            oldR5.compatibilityReason(machine, build, plan.signature()));

        try {
            KernelCalibrationProfile.fromJson(profile.toJson().replace(
                "\"R2_JAVA\"", "\"NOT_A_BACKEND\""));
            throw new AssertionError("unknown backend tag was accepted");
        } catch (IllegalArgumentException expected) {
            // The profile loader rejects before a stringly-typed choice can
            // reach runtime selection.
        }
    }

    @Test
    public void profileWinnerIsAnExactChoiceAndFallbackIsBaselineScalarThenJava() {
        PseudokernelPlan plan = lower(2, 5);
        GeneratedKernelRegistry registry = new GeneratedKernelRegistry();
        KernelVariantSignature scalarVariant = KernelVariantSignature.legacyScalar(plan.signature());
        KernelVariantSignature avxVariant = KernelVariantSignature.baselineAvx2(plan.signature());
        register(registry, plan, scalarVariant);
        register(registry, plan, avxVariant);
        RuntimeEnvironment allNative = environment(registry, true, true);

        KernelDispatchSelector cold = new KernelDispatchSelector(
            allNative, (KernelCalibrationProfile) null);
        assertTrue(cold.selectChoice(plan.signature()) instanceof GeneratedAvx2Backend);

        KernelDispatchSelector scalarOnly = new KernelDispatchSelector(
            environment(registry, true, false), (KernelCalibrationProfile) null);
        assertTrue(scalarOnly.selectChoice(plan.signature()) instanceof ScalarNativeBackend);
        assertEquals(scalarVariant,
            scalarOnly.selectChoice(plan.signature()).variantOptional().orElseThrow());

        KernelDispatchSelector javaOnly = new KernelDispatchSelector(
            environment(registry, false, false), (KernelCalibrationProfile) null);
        assertTrue(javaOnly.selectChoice(plan.signature()) instanceof R2JavaBackend);

        BackendCalibrationResult exactAvx = result(plan.signature(),
            new GeneratedAvx2Backend(avxVariant),
            new GeneratedAvx2Backend(avxVariant),
            List.of(pass(new GeneratedAvx2Backend(avxVariant), 80)));
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            testMachine(), testBuild(), exactAvx);
        KernelDispatchSelector exact = new KernelDispatchSelector(
            allNative, profile);
        KernelDispatchSelector.Selection selection = exact.select(plan.signature());
        assertTrue(selection.choice() instanceof GeneratedAvx2Backend);
        assertEquals(avxVariant, selection.choice().variantOptional().orElseThrow());
        assertNotNull(selection.entry());
        assertTrue(selection.profileHit());
    }

    @Test
    public void unavailableProfileChoiceFallsThroughWithoutInvokingIt() {
        PseudokernelPlan plan = lower(2, 5);
        GeneratedKernelRegistry registry = new GeneratedKernelRegistry();
        KernelVariantSignature avxVariant = KernelVariantSignature.baselineAvx2(plan.signature());
        register(registry, plan, avxVariant);
        BackendChoice avx = new GeneratedAvx2Backend(avxVariant);
        BackendCalibrationResult result = result(plan.signature(), avx, avx,
            List.of(pass(avx, 80)));
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            testMachine(), testBuild(), result);
        KernelDispatchSelector selector = new KernelDispatchSelector(
            environment(registry, true, false), profile);
        BackendChoice selected = selector.selectChoice(plan.signature());
        assertTrue(selected instanceof R2JavaBackend);
        assertFalse(selector.select(plan.signature()).profileHit());
    }

    @Test
    public void calibratorTimesOnlyCorrectnessGatedCandidatesAndSelectsMeasuredWinner() {
        PseudokernelPlan plan = lower(2, 5);
        BackendChoice java = new R2JavaBackend();
        BackendChoice scalar = new ScalarNativeBackend(
            KernelVariantSignature.legacyScalar(plan.signature()));
        BackendChoice avx = new GeneratedAvx2Backend(
            KernelVariantSignature.baselineAvx2(plan.signature()));
        AtomicInteger benchmarked = new AtomicInteger();
        BackendCalibrationResult result = BackendCalibrationRunner.calibrate(
            plan.signature(), "2x2/ops=5", avx,
            List.of(BackendCalibrationCandidate.valid(java),
                BackendCalibrationCandidate.valid(scalar),
                BackendCalibrationCandidate.valid(avx),
                BackendCalibrationCandidate.invalid(
                    new GeneratedAvx2Backend(new KernelVariantSignature(
                        plan.signature(), CodegenBackend.AVX2, "avx2", 4, 2,
                        KernelLoopForm.FLAT, KernelTailPolicy.SCALAR,
                        OptimizationSemantics.STRICT, FmaContraction.OFF)),
                    "correctness mismatch")),
            (choice, ignored) -> {
                benchmarked.incrementAndGet();
                long median = choice instanceof R2JavaBackend ? 130
                    : choice instanceof ScalarNativeBackend ? 100 : 60;
                return KernelBenchmarkStatistics.from(new long[] {
                    median - 1, median, median, median + 1, median
                });
            }, new KernelTuningConfig(0, 5, 10, 0L, 0.10, 0.03));

        assertTrue(result.winner() instanceof GeneratedAvx2Backend);
        assertEquals("baseline_avx2", result.winner().variantOptional()
            .orElseThrow().variantId());
        assertEquals(3, benchmarked.get());
        assertTrue(result.candidates().stream().anyMatch(candidate ->
            candidate.outcome() == KernelCandidateOutcome.CORRECTNESS_FAILED));
    }

    @Test
    public void calibratorCanPromoteJavaOrScalarPeerWhenTheirEvidenceWins() {
        PseudokernelPlan plan = lower(2, 5);
        BackendChoice java = new R2JavaBackend();
        BackendChoice scalar = new ScalarNativeBackend(
            KernelVariantSignature.legacyScalar(plan.signature()));
        BackendChoice avx = new GeneratedAvx2Backend(
            KernelVariantSignature.baselineAvx2(plan.signature()));
        List<BackendCalibrationCandidate> candidates = List.of(
            BackendCalibrationCandidate.valid(java),
            BackendCalibrationCandidate.valid(scalar),
            BackendCalibrationCandidate.valid(avx));
        KernelTuningConfig config = new KernelTuningConfig(0, 5, 10, 0L, 0.10, 0.03);

        BackendCalibrationResult javaResult = BackendCalibrationRunner.calibrate(
            plan.signature(), "2x2/ops=5", avx, candidates,
            (choice, ignored) -> statistics(choice instanceof R2JavaBackend ? 50 :
                choice instanceof ScalarNativeBackend ? 100 : 120), config);
        assertTrue(javaResult.winner() instanceof R2JavaBackend);

        BackendCalibrationResult scalarResult = BackendCalibrationRunner.calibrate(
            plan.signature(), "2x2/ops=5", avx, candidates,
            (choice, ignored) -> statistics(choice instanceof R2JavaBackend ? 130 :
                choice instanceof ScalarNativeBackend ? 60 : 100), config);
        assertTrue(scalarResult.winner() instanceof ScalarNativeBackend);
        assertEquals(scalar, scalarResult.winner());
    }

    @Test
    public void noStableWinnerRoundTripsWithoutReconstructingBaselineAsWinner() {
        PseudokernelPlan plan = lower(2, 5);
        BackendChoice java = new R2JavaBackend();
        BackendChoice avx = new GeneratedAvx2Backend(
            KernelVariantSignature.baselineAvx2(plan.signature()));
        KernelTuningConfig config = new KernelTuningConfig(0, 5, 10, 0L, 0.10, 0.03);
        BackendCalibrationResult result = BackendCalibrationRunner.calibrate(
            plan.signature(), "2x2/ops=5", avx,
            List.of(BackendCalibrationCandidate.valid(java),
                BackendCalibrationCandidate.valid(avx)),
            (choice, ignored) -> KernelBenchmarkStatistics.from(
                new long[] {1, 2, 100, 200, 400}), config);

        assertEquals(BackendSelectionStatus.NO_STABLE_WINNER, result.selectionStatus());
        assertNull(result.winner());
        assertFalse(result.stable());
        assertEquals(0.0, result.winnerSpeedup(), 0.0);

        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            testMachine(), testBuild(), result);
        KernelCalibrationProfile loaded = KernelCalibrationProfile.fromJson(profile.toJson());
        KernelCalibrationEntry entry = loaded.entry(plan.signature()).orElseThrow();
        assertEquals(BackendSelectionStatus.NO_STABLE_WINNER, entry.selectionStatus());
        assertEquals(avx, entry.baselineChoice());
        assertNull(entry.winnerChoice());
        assertFalse(entry.stable());
        assertTrue(profile.toJson().contains("\"winnerChoice\" : null"));

        GeneratedKernelRegistry registry = new GeneratedKernelRegistry();
        register(registry, plan, KernelVariantSignature.baselineAvx2(plan.signature()));
        KernelDispatchSelector selector = new KernelDispatchSelector(
            environment(registry, true, true), loaded);
        BackendChoice fallback = selector.selectChoice(plan.signature());
        assertTrue(fallback instanceof GeneratedAvx2Backend);
        assertFalse(selector.select(plan.signature()).profileHit());
        assertTrue(selector.select(plan.signature()).reason().contains("no stable winner"));
    }

    @Test
    public void profileMergeReplacesOneBucketAndPreservesOtherKernels() {
        PseudokernelPlan first = lower(2, 5);
        PseudokernelPlan second = lower(3, 7);
        BackendChoice firstChoice = new R2JavaBackend();
        BackendChoice secondChoice = new R2JavaBackend();
        BackendCalibrationResult firstResult = result(first.signature(), firstChoice,
            firstChoice, List.of(pass(firstChoice, 100)));
        BackendCalibrationResult secondResult = result(second.signature(), secondChoice,
            secondChoice, List.of(pass(secondChoice, 200)));
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            testMachine(), testBuild(), firstResult).merge(secondResult);
        assertEquals(2, profile.entries().size());

        BackendCalibrationResult replacement = result(first.signature(), firstChoice,
            firstChoice, List.of(pass(firstChoice, 50)));
        KernelCalibrationProfile updated = profile.merge(replacement);
        assertEquals(2, updated.entries().size());
        assertEquals(50.0, updated.entry(first.signature()).orElseThrow()
            .winnerChoice() instanceof R2JavaBackend
                ? updated.entry(first.signature()).orElseThrow().backendCandidates().get(0)
                    .statistics().medianNanos() : -1.0, 0.0);
        assertNotNull(updated.entry(second.signature()).orElse(null));
    }

    @Test
    public void unknownOrUnverifiedBuildIdentityCannotActivateAProfile() {
        PseudokernelPlan plan = lower(2, 5);
        BackendChoice java = new R2JavaBackend();
        BackendCalibrationResult result = result(plan.signature(), java, java,
            List.of(pass(java, 100)));
        KernelBuildIdentity unverified = new KernelBuildIdentity(
            "test", "unknown", "kernel-v1", "variant-v1", "unknown",
            KernelBuildIdentity.CODEGEN_ABI_VERSION, KernelBuildIdentity.STRICT_FLAGS,
            false, false, "unknown", "unknown", "unknown", "unknown");
        KernelCalibrationProfile profile = KernelCalibrationProfile.fromBackendResult(
            testMachine(), unverified, result);
        assertEquals("build identity unverified",
            profile.compatibilityReason(testMachine(), unverified, plan.signature()));
    }

    private static BackendCandidateEvidence pass(BackendChoice choice, long median) {
        return new BackendCandidateEvidence(choice, KernelCandidateOutcome.PASS,
            true, true, KernelBenchmarkStatistics.from(new long[] {
                median, median, median
            }), 1L, "test evidence");
    }

    private static KernelBenchmarkStatistics statistics(long median) {
        return KernelBenchmarkStatistics.from(new long[] {
            median, median, median, median, median
        });
    }

    private static BackendCalibrationResult result(KernelSignature signature,
                                                   BackendChoice baseline,
                                                   BackendChoice winner,
                                                   List<BackendCandidateEvidence> evidence) {
        return new BackendCalibrationResult(signature, "exact", baseline, evidence, winner,
            1.0, "test decision");
    }

    private static void register(GeneratedKernelRegistry registry,
                                 PseudokernelPlan plan,
                                 KernelVariantSignature variant) {
        GeneratedKernelSource source = KernelCodeGenerator.variant(plan, variant);
        registry.register(source.descriptor(), binding -> {
            KernelReferenceExecutor.executeVerified(plan.function().body() == null
                ? plan.function() : plan.function(), binding);
            return true;
        });
    }

    private static RuntimeEnvironment environment(GeneratedKernelRegistry registry,
                                                  boolean nativeAvailable,
                                                  boolean avx2) {
        return new RuntimeEnvironment(registry, testMachine(), testBuild(),
            true, nativeAvailable, avx2, avx2);
    }

    private static KernelMachineIdentity testMachine() {
        return new KernelMachineIdentity("x86_64", "test", "6", "1", "test-cpu",
            "avx,avx2,fma", "test-os", "test-jvm", "test-c++", 8);
    }

    private static KernelBuildIdentity testBuild() {
        return new KernelBuildIdentity("test", "test-sha", "kernel-v1", "variant-v1",
            "test-c++");
    }

    private static PseudokernelPlan lower(int size, int operations) {
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
        return PseudokernelPlanner.plan(KernelLowerer.lower(
            program.cpuPlan().fusedRegions().get(0)).program().function());
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + index * 0.001;
        }
        return Matrix.wrap(data, rows, columns);
    }
}
