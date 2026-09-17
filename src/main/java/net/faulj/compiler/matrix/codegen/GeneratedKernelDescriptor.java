package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

import net.faulj.compiler.matrix.kernel.KernelValueType;

/** Registry metadata for one compiled generated function. */
public final class GeneratedKernelDescriptor {
    private final KernelSignature signature;
    private final String symbol;
    private final CodegenBackend backend;
    private final String requiredCpuFeature;
    private final KernelValueType valueType;
    private final int vectorWidth;
    private final String sourceProvenance;
    private final int sourceSizeBytes;
    private final int maxLiveVectorValues;
    private final long rows;
    private final long columns;

    private GeneratedKernelDescriptor(KernelSignature signature,
                                      String symbol,
                                      CodegenBackend backend,
                                      String requiredCpuFeature,
                                      KernelValueType valueType,
                                      int vectorWidth,
                                      String sourceProvenance,
                                      int sourceSizeBytes,
                                      int maxLiveVectorValues,
                                      long rows,
                                      long columns) {
        this.signature = Objects.requireNonNull(signature, "Signature");
        this.symbol = Objects.requireNonNull(symbol, "Symbol");
        this.backend = Objects.requireNonNull(backend, "Backend");
        this.requiredCpuFeature = Objects.requireNonNull(requiredCpuFeature, "CPU feature");
        this.valueType = Objects.requireNonNull(valueType, "Value type");
        this.vectorWidth = vectorWidth;
        this.sourceProvenance = Objects.requireNonNull(sourceProvenance, "Source provenance");
        this.sourceSizeBytes = sourceSizeBytes;
        this.maxLiveVectorValues = maxLiveVectorValues;
        this.rows = rows;
        this.columns = columns;
    }

    public static GeneratedKernelDescriptor from(GeneratedKernelSource source) {
        if (source == null) {
            throw new IllegalArgumentException("Generated source must not be null");
        }
        return new GeneratedKernelDescriptor(
            source.signature(), source.symbol(), source.backend(),
            source.backend().requiredCpuFeature(), KernelValueType.FP64,
            source.plan().vectorWidth(), source.sourceProvenance(), source.sourceSizeBytes(),
            source.plan().maxLiveVectorValues(),
            source.plan().function().loops().get(0).upperBound()
                - source.plan().function().loops().get(0).lowerBound(),
            source.plan().function().loops().get(1).upperBound()
                - source.plan().function().loops().get(1).lowerBound());
    }

    public KernelSignature signature() {
        return signature;
    }

    public String symbol() {
        return symbol;
    }

    public CodegenBackend backend() {
        return backend;
    }

    public String requiredCpuFeature() {
        return requiredCpuFeature;
    }

    public KernelValueType valueType() {
        return valueType;
    }

    public int vectorWidth() {
        return vectorWidth;
    }

    public String sourceProvenance() {
        return sourceProvenance;
    }

    public int sourceSizeBytes() {
        return sourceSizeBytes;
    }

    public int maxLiveVectorValues() {
        return maxLiveVectorValues;
    }

    public long rows() {
        return rows;
    }

    public long columns() {
        return columns;
    }

    @Override
    public String toString() {
        return "generated-kernel " + signature.shortHash()
            + " symbol=" + symbol + " backend=" + backend
            + " isa=" + requiredCpuFeature + " source=" + sourceProvenance;
    }
}
