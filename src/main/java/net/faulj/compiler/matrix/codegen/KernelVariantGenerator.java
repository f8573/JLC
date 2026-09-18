package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.List;

import net.faulj.compiler.matrix.OptimizationSemantics;

/**
 * Deterministic, deliberately bounded R5 candidate generator.
 *
 * <p>It consumes only an already verified R4 plan. It never changes Kernel IR
 * and never invents aliasing or schedule legality facts.</p>
 */
public final class KernelVariantGenerator {
    public static final int DEFAULT_MAX_CANDIDATES = 10;
    public static final int[] AVX2_UNROLLS = {1, 2, 4, 8};
    public static final int MAX_PRACTICAL_REGISTER_ESTIMATE = 32;

    private KernelVariantGenerator() {
    }

    public static List<KernelVariantCandidate> enumerate(PseudokernelPlan plan) {
        return enumerate(plan, OptimizationSemantics.STRICT, DEFAULT_MAX_CANDIDATES);
    }

    /**
     * Enumerate with an explicit semantic permission. FMA is intentionally
     * not included unless the caller explicitly opts into it and the variant
     * family is requested through {@link #enumerateWithFma}.
     */
    public static List<KernelVariantCandidate> enumerate(PseudokernelPlan plan,
                                                          OptimizationSemantics semantics,
                                                          int maxCandidates) {
        requirePlan(plan);
        if (semantics == null) {
            throw new IllegalArgumentException("Optimization semantics must not be null");
        }
        if (maxCandidates < 1) {
            throw new IllegalArgumentException("maxCandidates must be positive");
        }

        List<KernelVariantCandidate> candidates = new ArrayList<>();
        if (plan.scalarCppEligible()) {
            addBounded(candidates, scalarCandidate(plan, KernelLoopForm.NESTED, semantics),
                maxCandidates);
            if (plan.avx2Eligible()) {
                addBounded(candidates, scalarCandidate(plan, KernelLoopForm.FLAT, semantics),
                    maxCandidates);
            }
        }
        if (plan.avx2Eligible()) {
            for (int unroll : AVX2_UNROLLS) {
                for (KernelLoopForm form : List.of(KernelLoopForm.FLAT, KernelLoopForm.NESTED)) {
                    KernelVariantSignature signature = new KernelVariantSignature(
                        plan.signature(), CodegenBackend.AVX2, "avx2", 4, unroll, form,
                        KernelTailPolicy.SCALAR, semantics, FmaContraction.OFF);
                    addBounded(candidates, candidate(plan, signature), maxCandidates);
                }
            }
        }
        if (plan.avx2Eligible()) {
            KernelVariantSignature baseline = KernelVariantSignature.baselineAvx2(plan.signature());
            if (candidates.stream().noneMatch(item -> item.signature().equals(baseline))) {
                KernelVariantCandidate baselineCandidate = candidate(plan, baseline);
                if (candidates.size() >= maxCandidates) {
                    candidates.set(Math.max(0, maxCandidates - 1), baselineCandidate);
                } else {
                    candidates.add(baselineCandidate);
                }
            }
        }
        return List.copyOf(candidates);
    }

    /**
     * Explicitly opt-in relaxed/fast FMA family. The default generator never
     * calls this method, which makes STRICT the safe boundary even when a
     * KernelFunction came from a compiler that omitted semantic metadata.
     */
    public static List<KernelVariantCandidate> enumerateWithFma(PseudokernelPlan plan,
                                                                OptimizationSemantics semantics,
                                                                int maxCandidates) {
        if (semantics == null || semantics == OptimizationSemantics.STRICT) {
            throw new IllegalArgumentException("Explicit FMA requires RELAXED or FAST semantics");
        }
        List<KernelVariantCandidate> base = enumerate(plan, semantics, maxCandidates);
        if (!plan.avx2Eligible() || base.size() >= maxCandidates) {
            return base;
        }
        List<KernelVariantCandidate> result = new ArrayList<>(base);
        for (int unroll : AVX2_UNROLLS) {
            for (KernelLoopForm form : List.of(KernelLoopForm.FLAT, KernelLoopForm.NESTED)) {
                if (result.size() >= maxCandidates) {
                    break;
                }
                KernelVariantSignature signature = new KernelVariantSignature(
                    plan.signature(), CodegenBackend.AVX2, "avx2+fma", 4, unroll, form,
                    KernelTailPolicy.SCALAR, semantics, FmaContraction.EXPLICIT);
                result.add(candidate(plan, signature));
            }
        }
        return List.copyOf(result);
    }

    private static KernelVariantCandidate scalarCandidate(PseudokernelPlan plan,
                                                           KernelLoopForm form,
                                                           OptimizationSemantics semantics) {
        return candidate(plan, new KernelVariantSignature(
            plan.signature(), CodegenBackend.SCALAR_CPP, "none", 1, 1, form,
            KernelTailPolicy.SCALAR, semantics, FmaContraction.OFF));
    }

    private static KernelVariantCandidate candidate(PseudokernelPlan plan,
                                                     KernelVariantSignature signature) {
        int pressure = estimatePressure(plan, signature);
        if (signature.backend() == CodegenBackend.AVX2
            && !RuntimeCpuFeatures.avx2Supported()) {
            return KernelVariantCandidate.pruned(signature, pressure,
                "required ISA unavailable on calibration host");
        }
        if (signature.fmaContraction() == FmaContraction.EXPLICIT
            && !RuntimeCpuFeatures.fmaSupported()) {
            return KernelVariantCandidate.pruned(signature, pressure,
                "required FMA ISA unavailable on calibration host");
        }
        if (signature.backend() == CodegenBackend.AVX2
            && plan.vectorIterations() > 0
            && signature.unroll() > plan.vectorIterations()) {
            return KernelVariantCandidate.pruned(signature, pressure,
                "unroll exceeds available vector iterations for exact shape");
        }
        if (signature.backend() == CodegenBackend.AVX2 && pressure > MAX_PRACTICAL_REGISTER_ESTIMATE) {
            return KernelVariantCandidate.pruned(signature, pressure,
                "estimated vector register pressure is structurally excessive");
        }
        if (signature.loopForm() == KernelLoopForm.FLAT && !plan.avx2Eligible()
            && signature.backend() == CodegenBackend.SCALAR_CPP) {
            return KernelVariantCandidate.pruned(signature, pressure,
                "flat traversal requires proven contiguous accesses");
        }
        return KernelVariantCandidate.eligible(signature, pressure);
    }

    /**
     * Conservative estimate: live SSA values replicated per vector instance,
     * hoisted constants, and a small address/control allowance. It is used as
     * a pruning explanation, not as a claim about allocator behavior.
     */
    public static int estimatePressure(PseudokernelPlan plan,
                                       KernelVariantSignature signature) {
        if (signature.backend() != CodegenBackend.AVX2) {
            return 0;
        }
        long estimate = (long) plan.maxLiveVectorValues() * signature.unroll()
            + plan.constantCount() + 2L;
        return estimate > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) estimate;
    }

    private static void addBounded(List<KernelVariantCandidate> candidates,
                                   KernelVariantCandidate candidate,
                                   int maxCandidates) {
        if (candidates.size() < maxCandidates) {
            candidates.add(candidate);
        }
    }

    private static void requirePlan(PseudokernelPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("Pseudokernel plan must not be null");
        }
    }
}
