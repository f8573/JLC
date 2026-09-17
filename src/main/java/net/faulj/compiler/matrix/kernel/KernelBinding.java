package net.faulj.compiler.matrix.kernel;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.cpu.PhysicalBufferSlot;
import net.faulj.compiler.matrix.cpu.PhysicalMemoryPlan;
import net.faulj.matrix.Matrix;

/**
 * Runtime binding kept outside immutable Kernel IR.
 *
 * <p>The optional physical-slot map makes the R1 boundary visible without
 * making a kernel own allocations or retain mutable matrices.</p>
 */
public final class KernelBinding {
    private final KernelFunction function;
    private final Map<KernelBuffer, Matrix> matrices;
    private final Map<KernelBuffer, PhysicalBufferSlot> physicalSlots;

    public KernelBinding(KernelFunction function,
                         Map<KernelBuffer, Matrix> matrices) {
        this(function, matrices, Map.of());
    }

    public KernelBinding(KernelFunction function,
                         Map<KernelBuffer, Matrix> matrices,
                         Map<KernelBuffer, PhysicalBufferSlot> physicalSlots) {
        if (function == null || matrices == null || physicalSlots == null) {
            throw new IllegalArgumentException("Kernel binding arguments must not be null");
        }
        IdentityHashMap<KernelBuffer, Matrix> matrixCopy = new IdentityHashMap<>();
        for (Map.Entry<KernelBuffer, Matrix> entry : matrices.entrySet()) {
            if (entry.getKey() == null) {
                throw new IllegalArgumentException("Kernel matrix binding buffer must not be null");
            }
            if (entry.getValue() != null) {
                matrixCopy.put(entry.getKey(), entry.getValue());
            }
        }
        IdentityHashMap<KernelBuffer, PhysicalBufferSlot> slotCopy = new IdentityHashMap<>();
        for (Map.Entry<KernelBuffer, PhysicalBufferSlot> entry : physicalSlots.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                throw new IllegalArgumentException("Kernel physical binding must not contain nulls");
            }
            slotCopy.put(entry.getKey(), entry.getValue());
        }
        this.function = function;
        this.matrices = Collections.unmodifiableMap(matrixCopy);
        this.physicalSlots = Collections.unmodifiableMap(slotCopy);
    }

    public static KernelBinding fromLogicalBuffers(
        KernelFunction function,
        Map<LogicalBuffer, Matrix> logicalMatrices) {
        return fromLogicalBuffers(function, logicalMatrices, null);
    }

    public static KernelBinding fromLogicalBuffers(
        KernelFunction function,
        Map<LogicalBuffer, Matrix> logicalMatrices,
        PhysicalMemoryPlan physicalMemoryPlan) {
        if (function == null || logicalMatrices == null) {
            throw new IllegalArgumentException("Kernel logical bindings must not be null");
        }
        IdentityHashMap<KernelBuffer, Matrix> matrices = new IdentityHashMap<>();
        IdentityHashMap<KernelBuffer, PhysicalBufferSlot> slots = new IdentityHashMap<>();
        for (KernelBuffer buffer : function.buffers()) {
            LogicalBuffer logical = buffer.logicalBuffer();
            if (logical != null && logicalMatrices.containsKey(logical)) {
                Matrix matrix = logicalMatrices.get(logical);
                if (matrix != null) {
                    matrices.put(buffer, matrix);
                }
            }
            if (logical != null && physicalMemoryPlan != null) {
                PhysicalBufferSlot slot = physicalMemoryPlan.slotFor(logical);
                if (slot != null) {
                    slots.put(buffer, slot);
                }
            }
        }
        return new KernelBinding(function, matrices, slots);
    }

    public KernelFunction function() {
        return function;
    }

    public Map<KernelBuffer, Matrix> matrices() {
        return matrices;
    }

    public Map<KernelBuffer, PhysicalBufferSlot> physicalSlots() {
        return physicalSlots;
    }

    public Matrix matrix(KernelBuffer buffer) {
        return matrices.get(buffer);
    }

    public PhysicalBufferSlot physicalSlot(KernelBuffer buffer) {
        return physicalSlots.get(buffer);
    }

    public boolean isResolved(KernelBuffer buffer) {
        return matrices.containsKey(buffer);
    }

    @Override
    public String toString() {
        return "kernel binding " + function.name() + " matrices=" + matrices.keySet()
            + " physicalSlots=" + physicalSlots.keySet();
    }
}
