package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** Compiled candidate plus timing/provenance metadata supplied by the harness. */
public final class CompiledKernelVariant {
    private final KernelVariantCandidate candidate;
    private final GeneratedKernelInvoker invoker;
    private final long sourceGenerationNanos;
    private final long compilationNanos;
    private final long compiledTextBytes;
    private final String diagnostics;

    public CompiledKernelVariant(KernelVariantCandidate candidate,
                                 GeneratedKernelInvoker invoker,
                                 long sourceGenerationNanos,
                                 long compilationNanos,
                                 long compiledTextBytes,
                                 String diagnostics) {
        this.candidate = Objects.requireNonNull(candidate, "Candidate");
        this.invoker = Objects.requireNonNull(invoker, "Invoker");
        this.sourceGenerationNanos = Math.max(0L, sourceGenerationNanos);
        this.compilationNanos = Math.max(0L, compilationNanos);
        this.compiledTextBytes = Math.max(0L, compiledTextBytes);
        this.diagnostics = diagnostics == null ? "" : diagnostics;
    }

    public static CompiledKernelVariant of(KernelVariantCandidate candidate,
                                           GeneratedKernelInvoker invoker) {
        return new CompiledKernelVariant(candidate, invoker, 0L, 0L, 0L, "");
    }

    public KernelVariantCandidate candidate() {
        return candidate;
    }

    public KernelVariantSignature variantSignature() {
        return candidate.signature();
    }

    public GeneratedKernelInvoker invoker() {
        return invoker;
    }

    public long sourceGenerationNanos() {
        return sourceGenerationNanos;
    }

    public long compilationNanos() {
        return compilationNanos;
    }

    public long compiledTextBytes() {
        return compiledTextBytes;
    }

    public String diagnostics() {
        return diagnostics;
    }
}
