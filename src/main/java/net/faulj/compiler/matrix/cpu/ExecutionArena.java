package net.faulj.compiler.matrix.cpu;

import java.util.IdentityHashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * Execution-scoped owner of the live matrices described by a physical plan.
 *
 * <p>A slot is activated for one logical value at a time. Releasing a logical
 * value only makes its slot available for the next non-overlapping lifetime;
 * the arena retains ownership until the invocation's final cleanup.</p>
 */
final class ExecutionArena {
    private final PhysicalMemoryPlan plan;
    private final IdentityHashMap<PhysicalBufferSlot, Matrix> allocations = new IdentityHashMap<>();
    private final IdentityHashMap<PhysicalBufferSlot, LogicalBuffer> active = new IdentityHashMap<>();
    private boolean closed;

    ExecutionArena(PhysicalMemoryPlan plan) {
        this.plan = plan;
    }

    Matrix activate(LogicalBuffer buffer) {
        if (closed) {
            throw new IllegalStateException("Execution arena is already closed");
        }
        PhysicalBufferSlot slot = plan.requireSlot(buffer);
        LogicalBuffer activeBuffer = active.get(slot);
        if (activeBuffer != null && activeBuffer != buffer) {
            throw new IllegalStateException(
                "Physical slot P" + slot.slotId() + " is active for logical buffer %"
                    + activeBuffer.id());
        }
        Matrix matrix = allocations.get(slot);
        if (matrix == null) {
            matrix = allocate(slot);
            allocations.put(slot, matrix);
        }
        active.put(slot, buffer);
        return matrix;
    }

    void release(LogicalBuffer buffer) {
        PhysicalBufferSlot slot = plan.requireSlot(buffer);
        LogicalBuffer activeBuffer = active.get(slot);
        if (activeBuffer == null) {
            return;
        }
        if (activeBuffer != buffer) {
            throw new IllegalStateException(
                "Logical buffer %" + buffer.id() + " does not own active slot P"
                    + slot.slotId());
        }
        active.remove(slot);
    }

    int allocatedSlotCount() {
        return allocations.size();
    }

    void closeExcept(Matrix transferredResult, Throwable failure) {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException cleanupFailure = null;
        IdentityHashMap<Matrix, Boolean> closedMatrices = new IdentityHashMap<>();
        for (Map.Entry<PhysicalBufferSlot, Matrix> entry : allocations.entrySet()) {
            Matrix matrix = entry.getValue();
            if (matrix == null || matrix == transferredResult
                || closedMatrices.put(matrix, Boolean.TRUE) != null
                || !(matrix instanceof OffHeapMatrix offHeap)) {
                continue;
            }
            try {
                offHeap.close();
            } catch (RuntimeException exception) {
                if (cleanupFailure == null) {
                    cleanupFailure = exception;
                } else {
                    cleanupFailure.addSuppressed(exception);
                }
            }
        }
        if (cleanupFailure != null) {
            if (failure != null) {
                failure.addSuppressed(cleanupFailure);
            } else {
                throw cleanupFailure;
            }
        }
    }

    private static Matrix allocate(PhysicalBufferSlot slot) {
        if (!slot.storageClass().isKnown()) {
            throw new IllegalStateException(
                "Cannot execute physical slot P" + slot.slotId()
                    + " without known storage compatibility: " + slot.storageClass());
        }
        Matrix matrix;
        if (slot.storageClass().memorySpace() == MemorySpace.OFF_HEAP) {
            OffHeapMatrix offHeap = null;
            try {
                offHeap = new OffHeapMatrix(slot.shape().rows(), slot.shape().columns());
                if (slot.storageClass().valueKind() == StorageValueKind.COMPLEX) {
                    offHeap.ensureImagData();
                }
                return offHeap;
            } catch (RuntimeException | Error exception) {
                if (offHeap != null) {
                    offHeap.close();
                }
                throw exception;
            }
        }
        matrix = new Matrix(slot.shape().rows(), slot.shape().columns());
        if (slot.storageClass().valueKind() == StorageValueKind.COMPLEX) {
            matrix.ensureImagData();
        }
        return matrix;
    }
}
