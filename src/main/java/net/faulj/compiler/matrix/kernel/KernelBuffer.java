package net.faulj.compiler.matrix.kernel;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.BufferKind;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;

/**
 * Immutable logical buffer descriptor at a kernel boundary.
 *
 * <p>The descriptor deliberately keeps a {@link LogicalBuffer} identity and
 * never stores a live {@code Matrix}. Physical allocation and binding happen
 * later through R1.</p>
 */
public final class KernelBuffer {
    private final int id;
    private final String name;
    private final LogicalBuffer logicalBuffer;
    private final KernelBufferRole role;
    private final KernelValueType storageType;
    private final KernelValueType computeType;
    private final KernelValueType accumulatorType;
    private final MatrixShape shape;
    private final KernelLayout layout;
    private final MemorySpace memorySpace;
    private final BufferOwnership ownership;

    public KernelBuffer(int id,
                        String name,
                        LogicalBuffer logicalBuffer,
                        KernelBufferRole role,
                        KernelValueType storageType,
                        KernelValueType computeType,
                        KernelValueType accumulatorType,
                        MatrixShape shape,
                        KernelLayout layout,
                        MemorySpace memorySpace,
                        BufferOwnership ownership) {
        this.id = id;
        this.name = name;
        this.logicalBuffer = logicalBuffer;
        this.role = role;
        this.storageType = storageType;
        this.computeType = computeType;
        this.accumulatorType = accumulatorType;
        this.shape = shape;
        this.layout = layout;
        this.memorySpace = memorySpace;
        this.ownership = ownership;
    }

    /** Build the R3 descriptor from an M2 logical buffer without a Matrix. */
    public static KernelBuffer fromLogical(LogicalBuffer logicalBuffer,
                                           KernelBufferRole role) {
        if (logicalBuffer == null || role == null) {
            throw new IllegalArgumentException("Kernel logical buffer and role are required");
        }
        KernelLayout layout = logicalBuffer.isSymbolic()
            ? KernelLayout.UNKNOWN : KernelLayout.ROW_MAJOR_DENSE;
        return new KernelBuffer(
            logicalBuffer.id(), logicalBuffer.name(), logicalBuffer, role,
            KernelValueType.FP64, KernelValueType.FP64, KernelValueType.FP64,
            logicalBuffer.shape(), layout, logicalBuffer.memorySpace(),
            logicalBuffer.ownership());
    }

    public int id() {
        return id;
    }

    public String name() {
        return name;
    }

    public LogicalBuffer logicalBuffer() {
        return logicalBuffer;
    }

    public LogicalBuffer source() {
        return logicalBuffer;
    }

    /** Logical storage class inherited from the M2 buffer descriptor. */
    public BufferKind storageClass() {
        return logicalBuffer == null ? null : logicalBuffer.kind();
    }

    public KernelBufferRole role() {
        return role;
    }

    public KernelValueType storageType() {
        return storageType;
    }

    public KernelValueType computeType() {
        return computeType;
    }

    public KernelValueType accumulatorType() {
        return accumulatorType;
    }

    public MatrixShape shape() {
        return shape;
    }

    public KernelLayout layout() {
        return layout;
    }

    public MemorySpace memorySpace() {
        return memorySpace;
    }

    public BufferOwnership ownership() {
        return ownership;
    }

    public boolean isInput() {
        return role == KernelBufferRole.INPUT;
    }

    public boolean isOutput() {
        return role == KernelBufferRole.OUTPUT;
    }

    @Override
    public String toString() {
        return "%" + id + " " + name + " " + role + " " + storageType
            + " " + layout + " " + memorySpace;
    }
}
