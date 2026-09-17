package net.faulj.compiler.matrix.codegen;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Objects;

import net.faulj.compiler.matrix.OptimizationSemantics;

/**
 * Canonical identity for one implementation of a verified R3 kernel.
 *
 * <p>The semantic kernel signature is kept as a component rather than being
 * modified. Variant options affect only backend code generation and registry
 * identity. Timings are deliberately absent from this value.</p>
 */
public final class KernelVariantSignature implements Comparable<KernelVariantSignature> {
    public static final int CURRENT_VERSION = 1;

    private final KernelSignature kernelSignature;
    private final CodegenBackend backend;
    private final String isa;
    private final int vectorWidth;
    private final int unroll;
    private final KernelLoopForm loopForm;
    private final KernelTailPolicy tailPolicy;
    private final OptimizationSemantics semantics;
    private final FmaContraction fmaContraction;
    private final String canonicalText;
    private final String sha256;
    private final int hashCode;

    public KernelVariantSignature(KernelSignature kernelSignature,
                                  CodegenBackend backend,
                                  String isa,
                                  int vectorWidth,
                                  int unroll,
                                  KernelLoopForm loopForm,
                                  KernelTailPolicy tailPolicy,
                                  OptimizationSemantics semantics,
                                  FmaContraction fmaContraction) {
        this.kernelSignature = Objects.requireNonNull(kernelSignature, "Kernel signature");
        this.backend = Objects.requireNonNull(backend, "Backend");
        this.isa = requireText(isa, "ISA");
        this.vectorWidth = vectorWidth;
        this.unroll = unroll;
        this.loopForm = Objects.requireNonNull(loopForm, "Loop form");
        this.tailPolicy = Objects.requireNonNull(tailPolicy, "Tail policy");
        this.semantics = Objects.requireNonNull(semantics, "Optimization semantics");
        this.fmaContraction = Objects.requireNonNull(fmaContraction, "FMA policy");
        validate();
        this.canonicalText = canonicalText();
        this.sha256 = sha256(canonicalText);
        this.hashCode = canonicalText.hashCode();
    }

    /** The fixed R4 vector implementation retained as the generated baseline. */
    public static KernelVariantSignature baselineAvx2(KernelSignature signature) {
        return new KernelVariantSignature(signature, CodegenBackend.AVX2, "avx2", 4, 1,
            KernelLoopForm.FLAT, KernelTailPolicy.SCALAR, OptimizationSemantics.STRICT,
            FmaContraction.OFF);
    }

    /** The conservative scalar nested-loop implementation. */
    public static KernelVariantSignature scalar(KernelSignature signature,
                                                KernelLoopForm loopForm) {
        return new KernelVariantSignature(signature, CodegenBackend.SCALAR_CPP, "none", 1, 1,
            loopForm, KernelTailPolicy.SCALAR, OptimizationSemantics.STRICT,
            FmaContraction.OFF);
    }

    /** Compatibility identity for the original R4 scalar source shape. */
    public static KernelVariantSignature legacyScalar(KernelSignature signature) {
        return scalar(signature, KernelLoopForm.NESTED);
    }

    public KernelSignature kernelSignature() {
        return kernelSignature;
    }

    /** Alias useful at APIs that call the semantic component simply signature. */
    public KernelSignature signature() {
        return kernelSignature;
    }

    public CodegenBackend backend() {
        return backend;
    }

    public String isa() {
        return isa;
    }

    public String requiredCpuFeature() {
        return isa;
    }

    public int vectorWidth() {
        return vectorWidth;
    }

    public int unroll() {
        return unroll;
    }

    public KernelLoopForm loopForm() {
        return loopForm;
    }

    public KernelTailPolicy tailPolicy() {
        return tailPolicy;
    }

    public OptimizationSemantics semantics() {
        return semantics;
    }

    public FmaContraction fmaContraction() {
        return fmaContraction;
    }

    public boolean isBaselineAvx2() {
        return backend == CodegenBackend.AVX2 && vectorWidth == 4 && unroll == 1
            && loopForm == KernelLoopForm.FLAT && tailPolicy == KernelTailPolicy.SCALAR
            && fmaContraction == FmaContraction.OFF && semantics == OptimizationSemantics.STRICT;
    }

    public String canonicalText() {
        return canonicalTextFor(kernelSignature, backend, isa, vectorWidth, unroll, loopForm,
            tailPolicy, semantics, fmaContraction);
    }

    public String sha256() {
        return sha256;
    }

    public String shortHash() {
        return sha256.substring(0, 12);
    }

    /** Stable human-readable ID used in reports and native symbol names. */
    public String variantId() {
        if (isBaselineAvx2()) {
            return "baseline_avx2";
        }
        StringBuilder result = new StringBuilder(backend.propertyValue());
        if (backend == CodegenBackend.AVX2) {
            result.append("_u").append(unroll);
        }
        result.append('_').append(loopForm.propertyValue());
        if (fmaContraction != FmaContraction.OFF) {
            result.append("_fma_").append(fmaContraction.propertyValue());
        }
        return result.toString();
    }

    /** Safe deterministic C++ symbol; benchmark values never participate. */
    public String generatedSymbol() {
        if (isBaselineAvx2()) {
            // Preserve the public R4 baseline symbol for existing generated artifacts.
            return kernelSignature.generatedSymbol();
        }
        return kernelSignature.generatedSymbol() + "_" + variantId() + "_" + shortHash();
    }

    @Override
    public int compareTo(KernelVariantSignature other) {
        return canonicalText.compareTo(other.canonicalText);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof KernelVariantSignature signature
            && canonicalText.equals(signature.canonicalText);
    }

    @Override
    public int hashCode() {
        return hashCode;
    }

    @Override
    public String toString() {
        return canonicalText;
    }

    private void validate() {
        if (vectorWidth <= 0 || unroll <= 0) {
            throw new IllegalArgumentException("Vector width and unroll must be positive");
        }
        if (backend == CodegenBackend.AVX2) {
            if (vectorWidth != 4 || !("avx2".equals(isa) || "avx2+fma".equals(isa))) {
                throw new IllegalArgumentException("AVX2 variants require 4 FP64 lanes and avx2 ISA");
            }
            if (unroll != 1 && unroll != 2 && unroll != 4 && unroll != 8) {
                throw new IllegalArgumentException("AVX2 unroll must be one of 1, 2, 4, or 8");
            }
        } else if (backend == CodegenBackend.SCALAR_CPP
            && (vectorWidth != 1 || unroll != 1 || !"none".equals(isa))) {
            throw new IllegalArgumentException("Scalar C++ variants require width 1, unroll 1, and no ISA");
        }
        if (fmaContraction == FmaContraction.EXPLICIT && semantics == OptimizationSemantics.STRICT) {
            throw new IllegalArgumentException("STRICT variants cannot request explicit FMA contraction");
        }
        if (fmaContraction == FmaContraction.EXPLICIT && backend != CodegenBackend.AVX2) {
            throw new IllegalArgumentException("Explicit FMA is only defined for AVX2 variants");
        }
        if (fmaContraction == FmaContraction.EXPLICIT && !"avx2+fma".equals(isa)) {
            throw new IllegalArgumentException("Explicit FMA variants require avx2+fma ISA");
        }
    }

    private static String canonicalTextFor(KernelSignature signature,
                                           CodegenBackend backend,
                                           String isa,
                                           int vectorWidth,
                                           int unroll,
                                           KernelLoopForm loopForm,
                                           KernelTailPolicy tailPolicy,
                                           OptimizationSemantics semantics,
                                           FmaContraction fmaContraction) {
        return "jlc-kernel-variant-v" + CURRENT_VERSION + "\n"
            + "kernel-sha256=" + signature.sha256() + "\n"
            + "backend=" + backend.name() + "\n"
            + "isa=" + isa.toLowerCase(Locale.ROOT) + "\n"
            + "vector-width=" + vectorWidth + "\n"
            + "unroll=" + unroll + "\n"
            + "loop-form=" + loopForm.name() + "\n"
            + "tail-policy=" + tailPolicy.name() + "\n"
            + "semantics=" + semantics.name() + "\n"
            + "fma-contraction=" + fmaContraction.name() + "\n"
            + "kernel-canonical-sha256=" + signature.sha256() + "\n";
    }

    private static String requireText(String value, String label) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(label + " must not be blank");
        }
        return value.trim();
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                .digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("JDK must provide SHA-256", impossible);
        }
    }
}
