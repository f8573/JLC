package net.faulj.compiler.matrix.codegen;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
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

    /**
     * Parse and validate the exact canonical identity persisted by R5.
     * Unknown, duplicated, missing, or contradictory fields are rejected.
     */
    public static KernelVariantSignature fromCanonicalText(KernelSignature signature,
                                                            String canonicalText) {
        Objects.requireNonNull(signature, "Kernel signature");
        if (canonicalText == null || canonicalText.isBlank()) {
            throw new IllegalArgumentException("Kernel variant signature text is empty");
        }
        String[] lines = canonicalText.split("\\n", -1);
        if (lines.length == 0 || !("jlc-kernel-variant-v" + CURRENT_VERSION).equals(lines[0])) {
            throw new IllegalArgumentException("Unsupported kernel variant signature version");
        }
        Map<String, String> fields = new HashMap<>();
        for (int index = 1; index < lines.length; index++) {
            String line = lines[index];
            if (line.isEmpty()) {
                if (index != lines.length - 1) {
                    throw new IllegalArgumentException("Blank field in kernel variant signature");
                }
                continue;
            }
            int separator = line.indexOf('=');
            if (separator <= 0) {
                throw new IllegalArgumentException("Malformed kernel variant field: " + line);
            }
            String key = line.substring(0, separator);
            String value = line.substring(separator + 1);
            if (fields.putIfAbsent(key, value) != null) {
                throw new IllegalArgumentException("Duplicate kernel variant field: " + key);
            }
        }
        requireField(fields, "kernel-sha256", signature.sha256());
        requireField(fields, "kernel-canonical-sha256", signature.sha256());
        CodegenBackend backend = enumValue(fields, "backend", CodegenBackend.class);
        String isa = requiredField(fields, "isa");
        int vectorWidth = integerField(fields, "vector-width");
        int unroll = integerField(fields, "unroll");
        KernelLoopForm loopForm = enumValue(fields, "loop-form", KernelLoopForm.class);
        KernelTailPolicy tailPolicy = enumValue(fields, "tail-policy", KernelTailPolicy.class);
        OptimizationSemantics semantics = enumValue(fields, "semantics", OptimizationSemantics.class);
        FmaContraction fma = enumValue(fields, "fma-contraction", FmaContraction.class);
        KernelVariantSignature parsed = new KernelVariantSignature(
            signature, backend, isa, vectorWidth, unroll, loopForm, tailPolicy, semantics, fma);
        if (!parsed.canonicalText().equals(canonicalText)) {
            throw new IllegalArgumentException("Kernel variant canonical text is not normalized");
        }
        return parsed;
    }

    /** Alias for callers that use parser terminology. */
    public static KernelVariantSignature parse(KernelSignature signature,
                                                String canonicalText) {
        return fromCanonicalText(signature, canonicalText);
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

    private static String requiredField(Map<String, String> fields, String key) {
        String value = fields.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing kernel variant field: " + key);
        }
        return value;
    }

    private static void requireField(Map<String, String> fields, String key, String expected) {
        String actual = requiredField(fields, key);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException("Kernel variant " + key + " does not match kernel");
        }
    }

    private static int integerField(Map<String, String> fields, String key) {
        try {
            return Integer.parseInt(requiredField(fields, key));
        } catch (NumberFormatException failure) {
            throw new IllegalArgumentException("Invalid integer kernel variant field: " + key,
                failure);
        }
    }

    private static <T extends Enum<T>> T enumValue(Map<String, String> fields,
                                                   String key,
                                                   Class<T> type) {
        try {
            return Enum.valueOf(type, requiredField(fields, key));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("Invalid kernel variant field: " + key, failure);
        }
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
