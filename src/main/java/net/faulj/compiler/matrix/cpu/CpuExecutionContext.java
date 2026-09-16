package net.faulj.compiler.matrix.cpu;

import java.util.IdentityHashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/** Package-private runtime state for one CPU plan invocation. */
final class CpuExecutionContext {
    private final IdentityHashMap<LogicalBuffer, Matrix> values = new IdentityHashMap<>();
    private final IdentityHashMap<LogicalBuffer, Boolean> owned = new IdentityHashMap<>();
    private final PhysicalMemoryPlan physicalMemoryPlan;
    private final ExecutionArena arena;

    /** Preserve the pre-R1 context for focused step and ownership tests. */
    CpuExecutionContext() {
        this(null);
    }

    CpuExecutionContext(PhysicalMemoryPlan physicalMemoryPlan) {
        this.physicalMemoryPlan = physicalMemoryPlan;
        this.arena = physicalMemoryPlan == null ? null : new ExecutionArena(physicalMemoryPlan);
    }

    void bind(LogicalBuffer buffer, Matrix matrix) {
        values.put(buffer, matrix);
    }

    void bindOwned(LogicalBuffer buffer, Matrix matrix) {
        values.put(buffer, matrix);
        owned.put(buffer, Boolean.TRUE);
    }

    Matrix value(LogicalBuffer buffer) {
        Matrix value = values.get(buffer);
        if (value == null) {
            throw new IllegalStateException(
                "CPU step requires an unmaterialized logical buffer %" + buffer.id());
        }
        return value;
    }

    Matrix allocate(LogicalBuffer buffer) {
        if (arena != null) {
            if (buffer == null || !buffer.isTemporary() || !arenaPlanHasSlot(buffer)) {
                throw new IllegalStateException(
                    "Only materialized owned temporaries may use physical storage");
            }
            Matrix existing = values.get(buffer);
            if (existing != null) {
                return existing;
            }
            Matrix created = arena.activate(buffer);
            values.put(buffer, created);
            return created;
        }
        Matrix existing = values.get(buffer);
        if (existing != null) {
            return existing;
        }
        Matrix created = new Matrix(buffer.shape().rows(), buffer.shape().columns());
        values.put(buffer, created);
        owned.put(buffer, Boolean.TRUE);
        return created;
    }

    boolean reusesStorage() {
        return arena != null;
    }

    /** Release logical reachability after the inclusive lifetime ends. */
    void releaseExpired(int stepId) {
        if (arena == null) {
            return;
        }
        for (ExecutionLifetime lifetime : physicalMemoryPlan.executionLifetimes()) {
            if (lifetime.liveThroughReturn() || lifetime.endStep() != stepId) {
                continue;
            }
            LogicalBuffer buffer = lifetime.logicalBuffer();
            if (values.remove(buffer) != null) {
                arena.release(buffer);
            }
        }
    }

    Map<LogicalBuffer, Matrix> values() {
        return values;
    }

    boolean isOwned(LogicalBuffer buffer) {
        return owned.containsKey(buffer)
            || (arena != null && arenaPlanHasSlot(buffer));
    }

    void closeOwnedExcept(Matrix transferredResult, Throwable failure) {
        if (arena != null) {
            arena.closeExcept(transferredResult, failure);
            return;
        }
        IdentityHashMap<Matrix, Boolean> closed = new IdentityHashMap<>();
        RuntimeException cleanupFailure = null;
        for (Map.Entry<LogicalBuffer, Boolean> entry : owned.entrySet()) {
            Matrix matrix = values.get(entry.getKey());
            if (matrix == null || matrix == transferredResult
                || closed.put(matrix, Boolean.TRUE) != null
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

    private boolean arenaPlanHasSlot(LogicalBuffer buffer) {
        return buffer != null && physicalMemoryPlan.hasSlot(buffer);
    }
}
