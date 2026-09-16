package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.matrix.Matrix;

/**
 * R1 linear-scan physical storage planner.
 *
 * <p>This planner consumes the final {@link CpuStep} sequence. M2 statement
 * lifetimes are intentionally not consulted for executable reuse because M3
 * and M4 can reorder, fuse, tile, collapse, or elide those statements.</p>
 */
public final class PhysicalMemoryPlanner {
    private PhysicalMemoryPlanner() {
    }

    /** Build a reuse plan using the matrices captured by the CPU plan. */
    public static PhysicalMemoryPlan plan(CpuExecutionPlan cpuPlan) {
        if (cpuPlan == null) {
            throw new IllegalArgumentException("CPU execution plan must not be null");
        }
        return plan(cpuPlan, capturedBindings(cpuPlan));
    }

    /**
     * Build a reuse plan using a resolved set of borrowed external inputs.
     * This overload lets a plan-only lowering be safely reclassified at
     * execution time without mutating the immutable plan or its diagnostics.
     */
    static PhysicalMemoryPlan plan(CpuExecutionPlan cpuPlan,
                                   Map<LogicalBuffer, Matrix> runtimeBindings) {
        if (cpuPlan == null) {
            throw new IllegalArgumentException("CPU execution plan must not be null");
        }
        if (runtimeBindings == null) {
            throw new IllegalArgumentException("Runtime bindings must not be null");
        }

        IdentityHashMap<LogicalBuffer, PhysicalStorageClass> storage =
            storageProfiles(cpuPlan, runtimeBindings);
        List<ExecutionLifetime> lifetimes = executableLifetimes(cpuPlan);
        int logicalTemporaryCount = cpuPlan.temporaryCount();
        int elidedCount = cpuPlan.elidedTemporaryCount();

        long logicalBytes = 0L;
        for (ExecutionLifetime lifetime : lifetimes) {
            PhysicalStorageClass storageClass = requireStorage(
                storage, lifetime.logicalBuffer());
            logicalBytes = addSaturated(
                logicalBytes, storageClass.payloadBytes(lifetime.logicalBuffer().shape()));
        }

        List<SlotBuilder> slotBuilders = linearScan(lifetimes, storage);
        IdentityHashMap<LogicalBuffer, PhysicalBufferSlot> assignments = new IdentityHashMap<>();
        List<PhysicalBufferSlot> slots = new ArrayList<>(slotBuilders.size());
        for (SlotBuilder builder : slotBuilders) {
            PhysicalBufferSlot slot = new PhysicalBufferSlot(
                cpuPlan, builder.id, builder.storageClass, builder.shape, builder.logicalBuffers);
            slots.add(slot);
            for (LogicalBuffer buffer : builder.logicalBuffers) {
                if (assignments.put(buffer, slot) != null) {
                    throw new IllegalStateException(
                        "Logical buffer received multiple physical assignments");
                }
            }
        }

        long physicalBytes = 0L;
        for (PhysicalBufferSlot slot : slots) {
            physicalBytes = addSaturated(physicalBytes, slot.payloadBytes());
        }
        long peakLiveBytes = peakLiveBytes(lifetimes, storage, cpuPlan.stepCount());
        int peakConcurrentSlots = peakConcurrentSlots(lifetimes, cpuPlan.stepCount());
        return new PhysicalMemoryPlan(
            cpuPlan,
            lifetimes,
            slots,
            assignments,
            logicalTemporaryCount,
            elidedCount,
            logicalBytes,
            peakLiveBytes,
            physicalBytes,
            peakConcurrentSlots);
    }

    private static IdentityHashMap<LogicalBuffer, Matrix> capturedBindings(CpuExecutionPlan cpuPlan) {
        IdentityHashMap<LogicalBuffer, Matrix> result = new IdentityHashMap<>();
        for (CpuBufferBinding binding : cpuPlan.inputBindings()) {
            if (binding.matrix() != null) {
                result.put(binding.buffer(), binding.matrix());
            }
        }
        return result;
    }

    private static IdentityHashMap<LogicalBuffer, PhysicalStorageClass> storageProfiles(
        CpuExecutionPlan cpuPlan,
        Map<LogicalBuffer, Matrix> runtimeBindings) {
        IdentityHashMap<LogicalBuffer, PhysicalStorageClass> result = new IdentityHashMap<>();
        for (CpuBufferBinding binding : cpuPlan.inputBindings()) {
            Matrix matrix = runtimeBindings.get(binding.buffer());
            if (matrix != null) {
                validateShape(binding.buffer(), matrix);
                result.put(binding.buffer(), PhysicalStorageClass.fromMatrix(matrix));
            } else {
                result.put(binding.buffer(), new PhysicalStorageClass(
                    binding.buffer().memorySpace(), StorageValueKind.UNKNOWN));
            }
        }

        // Plan-only lowerings can contain symbolic inputs without a
        // CpuBufferBinding. Keep those plans inspectable; CpuExecutor rejects
        // them before any arena allocation unless a concrete binding exists.
        for (LogicalBuffer buffer : cpuPlan.affineProgram().buffers()) {
            if (buffer.isSymbolic() && !result.containsKey(buffer)) {
                result.put(buffer, new PhysicalStorageClass(
                    MemorySpace.UNKNOWN, StorageValueKind.UNKNOWN));
            }
        }

        for (CpuStep step : cpuPlan.steps()) {
            PhysicalStorageClass outputClass;
            if (step instanceof CpuGemmStep gemm) {
                PhysicalStorageClass lhs = requireStorage(result, gemm.lhs());
                PhysicalStorageClass rhs = requireStorage(result, gemm.rhs());
                outputClass = new PhysicalStorageClass(
                    gemmMemorySpace(lhs.memorySpace(), rhs.memorySpace()),
                    StorageValueKind.merge(lhs.valueKind(), rhs.valueKind()));
            } else {
                StorageValueKind valueKind = StorageValueKind.REAL;
                for (LogicalBuffer input : step.inputBuffers()) {
                    valueKind = StorageValueKind.merge(
                        valueKind, requireStorage(result, input).valueKind());
                }
                // The pre-R1 elementwise and transpose implementations create
                // ordinary Matrix results. Preserve that representation even
                // when an input is off-heap.
                outputClass = new PhysicalStorageClass(MemorySpace.HEAP, valueKind);
            }
            LogicalBuffer output = step.outputBuffer();
            if (output != null) {
                result.put(output, outputClass);
            }
        }
        return result;
    }

    private static List<ExecutionLifetime> executableLifetimes(CpuExecutionPlan cpuPlan) {
        IdentityHashMap<LogicalBuffer, Integer> producers = new IdentityHashMap<>();
        for (int index = 0; index < cpuPlan.steps().size(); index++) {
            CpuStep step = cpuPlan.steps().get(index);
            LogicalBuffer output = step.outputBuffer();
            if (output == null || !output.isTemporary()) {
                continue;
            }
            if (producers.put(output, index) != null) {
                throw new IllegalArgumentException(
                    "Logical buffer %" + output.id() + " has multiple CPU producers");
            }
        }

        List<ExecutionLifetime> result = new ArrayList<>();
        for (LogicalBuffer buffer : cpuPlan.materializedTemporaryBuffers()) {
            Integer producer = producers.get(buffer);
            if (producer == null) {
                throw new IllegalArgumentException(
                    "Materialized logical buffer %" + buffer.id() + " has no CPU producer");
            }
            Integer firstConsumer = null;
            Integer lastConsumer = null;
            for (int index = 0; index < cpuPlan.steps().size(); index++) {
                CpuStep step = cpuPlan.steps().get(index);
                for (LogicalBuffer input : step.inputBuffers()) {
                    if (input != buffer) {
                        continue;
                    }
                    if (index < producer) {
                        throw new IllegalArgumentException(
                            "Logical buffer %" + buffer.id()
                                + " is consumed before its CPU producer");
                    }
                    if (firstConsumer == null) {
                        firstConsumer = index;
                    }
                    lastConsumer = index;
                }
            }
            boolean liveThroughReturn = buffer == cpuPlan.outputBuffer();
            int endStep;
            if (liveThroughReturn) {
                endStep = Math.max(producer, cpuPlan.stepCount() - 1);
            } else {
                endStep = lastConsumer == null ? producer : lastConsumer;
            }
            result.add(new ExecutionLifetime(
                buffer, producer, firstConsumer, lastConsumer, liveThroughReturn, endStep));
        }
        return result;
    }

    private static List<SlotBuilder> linearScan(
        List<ExecutionLifetime> lifetimes,
        IdentityHashMap<LogicalBuffer, PhysicalStorageClass> storage) {
        List<ExecutionLifetime> sorted = new ArrayList<>(lifetimes);
        sorted.sort(Comparator.comparingInt(ExecutionLifetime::startStep)
            .thenComparingInt(lifetime -> lifetime.logicalBuffer().id()));
        List<SlotBuilder> slots = new ArrayList<>();
        List<ActiveAssignment> active = new ArrayList<>();
        for (ExecutionLifetime lifetime : sorted) {
            active.removeIf(assignment ->
                assignment.lifetime().endStep() < lifetime.startStep());
            PhysicalStorageClass storageClass = requireStorage(storage, lifetime.logicalBuffer());
            SlotBuilder selected = null;
            for (SlotBuilder candidate : slots) {
                if (isActive(candidate, active)
                    || !candidate.storageClass.compatibleWith(storageClass)
                    || !candidate.shape.equals(lifetime.logicalBuffer().shape())) {
                    continue;
                }
                selected = candidate;
                break;
            }
            if (selected == null) {
                selected = new SlotBuilder(
                    slots.size(), storageClass, lifetime.logicalBuffer().shape());
                slots.add(selected);
            }
            selected.logicalBuffers.add(lifetime.logicalBuffer());
            active.add(new ActiveAssignment(lifetime, selected));
        }
        return slots;
    }

    private static boolean isActive(SlotBuilder slot, List<ActiveAssignment> active) {
        return active.stream().anyMatch(assignment -> assignment.slot() == slot);
    }

    private static long peakLiveBytes(List<ExecutionLifetime> lifetimes,
                                      IdentityHashMap<LogicalBuffer, PhysicalStorageClass> storage,
                                      int stepCount) {
        long peak = 0L;
        for (int step = 0; step < stepCount; step++) {
            long current = 0L;
            for (ExecutionLifetime lifetime : lifetimes) {
                if (lifetime.containsStep(step)) {
                    current = addSaturated(current, requireStorage(
                        storage, lifetime.logicalBuffer())
                        .payloadBytes(lifetime.logicalBuffer().shape()));
                }
            }
            peak = Math.max(peak, current);
        }
        return peak;
    }

    private static int peakConcurrentSlots(List<ExecutionLifetime> lifetimes, int stepCount) {
        int peak = 0;
        for (int step = 0; step < stepCount; step++) {
            int current = 0;
            for (ExecutionLifetime lifetime : lifetimes) {
                if (lifetime.containsStep(step)) {
                    current++;
                }
            }
            peak = Math.max(peak, current);
        }
        return peak;
    }

    private static PhysicalStorageClass requireStorage(
        IdentityHashMap<LogicalBuffer, PhysicalStorageClass> storage,
        LogicalBuffer buffer) {
        PhysicalStorageClass result = storage.get(buffer);
        if (result == null) {
            throw new IllegalArgumentException(
                "No runtime storage classification for logical buffer %"
                    + (buffer == null ? "null" : buffer.id()));
        }
        return result;
    }

    private static MemorySpace gemmMemorySpace(MemorySpace lhs, MemorySpace rhs) {
        if (lhs == MemorySpace.OFF_HEAP || rhs == MemorySpace.OFF_HEAP) {
            return MemorySpace.OFF_HEAP;
        }
        if (lhs == MemorySpace.UNKNOWN || rhs == MemorySpace.UNKNOWN) {
            return MemorySpace.UNKNOWN;
        }
        return MemorySpace.HEAP;
    }

    private static void validateShape(LogicalBuffer buffer, Matrix matrix) {
        MatrixShape shape = buffer.shape();
        if (matrix.getRowCount() != shape.rows() || matrix.getColumnCount() != shape.columns()) {
            throw new IllegalArgumentException(
                "Runtime matrix shape " + matrix.getRowCount() + "x" + matrix.getColumnCount()
                    + " does not match logical buffer %" + buffer.id() + " shape " + shape);
        }
    }

    private static long addSaturated(long first, long second) {
        if (first == Long.MAX_VALUE || second == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        try {
            return Math.addExact(first, second);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    private static final class SlotBuilder {
        private final int id;
        private final PhysicalStorageClass storageClass;
        private final MatrixShape shape;
        private final List<LogicalBuffer> logicalBuffers = new ArrayList<>();

        private SlotBuilder(int id, PhysicalStorageClass storageClass, MatrixShape shape) {
            this.id = id;
            this.storageClass = storageClass;
            this.shape = shape;
        }
    }

    private record ActiveAssignment(ExecutionLifetime lifetime, SlotBuilder slot) {
    }
}
