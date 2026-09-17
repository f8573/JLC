package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.LogicalBuffer;

/**
 * One execution-owned physical storage slot in an R1 memory plan.
 *
 * <p>The slot is a plan description, not the live {@code Matrix} object. A
 * single execution arena activates the slot for each assigned logical value in
 * turn and owns the resulting storage until execution cleanup.</p>
 */
public final class PhysicalBufferSlot {
    private final CpuExecutionPlan owningPlan;
    private final int slotId;
    private final PhysicalStorageClass storageClass;
    private final MatrixShape shape;
    private final long payloadBytes;
    private final List<LogicalBuffer> logicalBuffers;

    PhysicalBufferSlot(CpuExecutionPlan owningPlan,
                       int slotId,
                       PhysicalStorageClass storageClass,
                       MatrixShape shape,
                       List<LogicalBuffer> logicalBuffers) {
        this.owningPlan = Objects.requireNonNull(
            owningPlan, "Physical slot owning plan must not be null");
        if (slotId < 0) {
            throw new IllegalArgumentException("Physical slot ID must be non-negative");
        }
        this.slotId = slotId;
        this.storageClass = Objects.requireNonNull(
            storageClass, "Physical slot storage class must not be null");
        this.shape = Objects.requireNonNull(shape, "Physical slot shape must not be null");
        if (logicalBuffers == null || logicalBuffers.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Physical slot logical buffers must not contain nulls");
        }
        this.logicalBuffers = Collections.unmodifiableList(new ArrayList<>(logicalBuffers));
        this.payloadBytes = storageClass.payloadBytes(shape);
    }

    public CpuExecutionPlan owningPlan() {
        return owningPlan;
    }

    public CpuExecutionPlan executionPlan() {
        return owningPlan;
    }

    public int slotId() {
        return slotId;
    }

    public int id() {
        return slotId;
    }

    public String displayId() {
        return "P" + slotId;
    }

    public PhysicalStorageClass storageClass() {
        return storageClass;
    }

    public PhysicalStorageClass compatibilityKey() {
        return storageClass;
    }

    public MatrixShape shape() {
        return shape;
    }

    public long capacityElements() {
        try {
            return Math.multiplyExact((long) shape.rows(), (long) shape.columns());
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    public long payloadBytes() {
        return payloadBytes;
    }

    public long bytes() {
        return payloadBytes;
    }

    public List<LogicalBuffer> logicalBuffers() {
        return logicalBuffers;
    }

    public List<LogicalBuffer> assignedLogicalBuffers() {
        return logicalBuffers;
    }

    public int logicalBufferCount() {
        return logicalBuffers.size();
    }

    public int reuseCount() {
        return Math.max(0, logicalBuffers.size() - 1);
    }

    public boolean isReused() {
        return logicalBuffers.size() > 1;
    }

    @Override
    public String toString() {
        return displayId() + " " + storageClass + " " + shape
            + " bytes=" + payloadBytes;
    }
}
