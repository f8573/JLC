package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

/** Immutable generated C++ translation unit and its backend metadata. */
public final class GeneratedKernelSource {
    private final PseudokernelPlan plan;
    private final CodegenBackend backend;
    private final String symbol;
    private final String source;
    private final String sourceProvenance;
    private final long generationTimeNanos;

    GeneratedKernelSource(PseudokernelPlan plan,
                          CodegenBackend backend,
                          String source,
                          CppEmissionOptions options,
                          long generationTimeNanos) {
        this.plan = Objects.requireNonNull(plan, "Pseudokernel plan");
        this.backend = Objects.requireNonNull(backend, "Codegen backend");
        this.symbol = plan.signature().generatedSymbol();
        this.source = Objects.requireNonNull(source, "Generated source");
        this.sourceProvenance = Objects.requireNonNull(options, "Emission options").sourceProvenance();
        this.generationTimeNanos = Math.max(0L, generationTimeNanos);
    }

    public PseudokernelPlan plan() {
        return plan;
    }

    public KernelSignature signature() {
        return plan.signature();
    }

    public CodegenBackend backend() {
        return backend;
    }

    public String symbol() {
        return symbol;
    }

    public String source() {
        return source;
    }

    public int sourceSizeBytes() {
        return source.getBytes(java.nio.charset.StandardCharsets.UTF_8).length;
    }

    public String sourceProvenance() {
        return sourceProvenance;
    }

    public long generationTimeNanos() {
        return generationTimeNanos;
    }

    public long generationTimeMicros() {
        return generationTimeNanos / 1_000L;
    }

    public GeneratedKernelDescriptor descriptor() {
        return GeneratedKernelDescriptor.from(this);
    }
}
