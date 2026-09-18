package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.LogicalBuffer;

/**
 * Immutable, inspectable physical storage plan for one lowered CPU plan.
 *
 * <p>The plan separates logical values and their executable step intervals
 * from reusable physical slots. It has no process-global state and does not
 * retain live matrix allocations between executions.</p>
 */
public final class PhysicalMemoryPlan {
    private final CpuExecutionPlan owningPlan;
    private final List<ExecutionLifetime> lifetimes;
    private final List<PhysicalBufferSlot> slots;
    private final IdentityHashMap<LogicalBuffer, ExecutionLifetime> lifetimesByBuffer;
    private final IdentityHashMap<LogicalBuffer, PhysicalBufferSlot> slotsByBuffer;
    private final int logicalTemporaryCount;
    private final int materializedLogicalTemporaryCount;
    private final int elidedLogicalTemporaryCount;
    private final long logicalTemporaryBytesSum;
    private final long peakLiveTemporaryBytes;
    private final long physicalTemporaryBytes;
    private final int logicalToPhysicalReuseCount;
    private final int peakConcurrentSlots;
    private final int reusedSlotCount;

    PhysicalMemoryPlan(CpuExecutionPlan owningPlan,
                       List<ExecutionLifetime> lifetimes,
                       List<PhysicalBufferSlot> slots,
                       Map<LogicalBuffer, PhysicalBufferSlot> slotsByBuffer,
                       int logicalTemporaryCount,
                       int elidedLogicalTemporaryCount,
                       long logicalTemporaryBytesSum,
                       long peakLiveTemporaryBytes,
                       long physicalTemporaryBytes,
                       int peakConcurrentSlots) {
        this.owningPlan = Objects.requireNonNull(
            owningPlan, "Physical memory plan owner must not be null");
        this.lifetimes = immutableCopy(lifetimes, "Execution lifetimes");
        this.slots = immutableCopy(slots, "Physical slots");
        this.lifetimesByBuffer = new IdentityHashMap<>();
        for (ExecutionLifetime lifetime : this.lifetimes) {
            if (lifetimesByBuffer.put(lifetime.logicalBuffer(), lifetime) != null) {
                throw new IllegalArgumentException("A logical buffer has multiple executable lifetimes");
            }
        }
        this.slotsByBuffer = new IdentityHashMap<>();
        if (slotsByBuffer == null) {
            throw new IllegalArgumentException("Physical slot assignments must not be null");
        }
        for (Map.Entry<LogicalBuffer, PhysicalBufferSlot> entry : slotsByBuffer.entrySet()) {
            LogicalBuffer buffer = Objects.requireNonNull(
                entry.getKey(), "Physical assignment buffer must not be null");
            PhysicalBufferSlot slot = Objects.requireNonNull(
                entry.getValue(), "Physical assignment slot must not be null");
            if (this.slotsByBuffer.put(buffer, slot) != null) {
                throw new IllegalArgumentException("A logical buffer has multiple physical slots");
            }
            if (!this.lifetimesByBuffer.containsKey(buffer)) {
                throw new IllegalArgumentException("Physical assignment has no executable lifetime");
            }
            if (!this.slots.contains(slot)) {
                throw new IllegalArgumentException("Physical assignment references an unknown slot");
            }
        }
        this.logicalTemporaryCount = requireNonNegative(
            logicalTemporaryCount, "Logical temporary count");
        this.materializedLogicalTemporaryCount = this.lifetimes.size();
        this.elidedLogicalTemporaryCount = requireNonNegative(
            elidedLogicalTemporaryCount, "Elided logical temporary count");
        if (this.materializedLogicalTemporaryCount + this.elidedLogicalTemporaryCount
            != this.logicalTemporaryCount) {
            throw new IllegalArgumentException(
                "Materialized and elided temporary counts do not match the logical count");
        }
        this.logicalTemporaryBytesSum = requireNonNegative(
            logicalTemporaryBytesSum, "Logical temporary bytes");
        this.peakLiveTemporaryBytes = requireNonNegative(
            peakLiveTemporaryBytes, "Peak live temporary bytes");
        this.physicalTemporaryBytes = requireNonNegative(
            physicalTemporaryBytes, "Physical temporary bytes");
        this.logicalToPhysicalReuseCount = Math.max(
            0, this.materializedLogicalTemporaryCount - this.slots.size());
        this.peakConcurrentSlots = requireNonNegative(
            peakConcurrentSlots, "Peak concurrent slots");
        this.reusedSlotCount = (int) this.slots.stream().filter(PhysicalBufferSlot::isReused).count();
        validateAssignments();
    }

    public CpuExecutionPlan owningPlan() {
        return owningPlan;
    }

    public CpuExecutionPlan executionPlan() {
        return owningPlan;
    }

    public List<ExecutionLifetime> executionLifetimes() {
        return lifetimes;
    }

    public List<ExecutionLifetime> lifetimes() {
        return lifetimes;
    }

    public ExecutionLifetime lifetime(LogicalBuffer buffer) {
        return lifetimesByBuffer.get(buffer);
    }

    public List<PhysicalBufferSlot> physicalSlots() {
        return slots;
    }

    public List<PhysicalBufferSlot> slots() {
        return slots;
    }

    /** Returns null for borrowed, symbolic, or fusion-elided buffers. */
    public PhysicalBufferSlot slotFor(LogicalBuffer buffer) {
        return slotsByBuffer.get(buffer);
    }

    public PhysicalBufferSlot physicalSlot(LogicalBuffer buffer) {
        return slotFor(buffer);
    }

    public boolean hasSlot(LogicalBuffer buffer) {
        return slotsByBuffer.containsKey(buffer);
    }

    public PhysicalBufferSlot requireSlot(LogicalBuffer buffer) {
        PhysicalBufferSlot slot = slotFor(buffer);
        if (slot == null) {
            throw new IllegalArgumentException(
                "Logical buffer has no physical slot: "
                    + (buffer == null ? "null" : "%" + buffer.id()));
        }
        return slot;
    }

    /** Inclusive interval interference query for two materialized values. */
    public boolean interferes(LogicalBuffer first, LogicalBuffer second) {
        if (first == null || second == null || first == second) {
            return false;
        }
        return interferes(lifetimesByBuffer.get(first), lifetimesByBuffer.get(second));
    }

    public boolean interferes(ExecutionLifetime first, ExecutionLifetime second) {
        if (first == null || second == null || first == second) {
            return false;
        }
        return first.startStep() <= second.endStep()
            && second.startStep() <= first.endStep();
    }

    public List<LogicalBuffer> interferingBuffers(LogicalBuffer buffer) {
        ExecutionLifetime target = lifetimesByBuffer.get(buffer);
        if (target == null) {
            return List.of();
        }
        List<LogicalBuffer> result = new ArrayList<>();
        for (ExecutionLifetime candidate : lifetimes) {
            if (candidate.logicalBuffer() != buffer && interferes(target, candidate)) {
                result.add(candidate.logicalBuffer());
            }
        }
        return Collections.unmodifiableList(result);
    }

    public int logicalTemporaryCount() {
        return logicalTemporaryCount;
    }

    public int materializedLogicalTemporaryCount() {
        return materializedLogicalTemporaryCount;
    }

    public int elidedLogicalTemporaryCount() {
        return elidedLogicalTemporaryCount;
    }

    public long logicalTemporaryBytesSum() {
        return logicalTemporaryBytesSum;
    }

    public long materializedLogicalTemporaryBytesSum() {
        return logicalTemporaryBytesSum;
    }

    public long peakLiveTemporaryBytes() {
        return peakLiveTemporaryBytes;
    }

    public long peakLiveLogicalBytes() {
        return peakLiveTemporaryBytes;
    }

    public long physicalTemporaryBytes() {
        return physicalTemporaryBytes;
    }

    public long physicalAllocatedBytes() {
        return physicalTemporaryBytes;
    }

    public int physicalSlotCount() {
        return slots.size();
    }

    public int logicalToPhysicalReuseCount() {
        return logicalToPhysicalReuseCount;
    }

    public int reusedSlotCount() {
        return reusedSlotCount;
    }

    public int peakConcurrentSlots() {
        return peakConcurrentSlots;
    }

    /** Number of slot allocations in the execution-scoped reuse arena. */
    public int allocationCount() {
        return slots.size();
    }

    /** A useful A/B baseline: one payload allocation for every materialized value. */
    public int legacyAllocationCount() {
        return materializedLogicalTemporaryCount;
    }

    public long legacyTemporaryBytes() {
        return logicalTemporaryBytesSum;
    }

    public long bytesAvoided() {
        if (logicalTemporaryBytesSum == Long.MAX_VALUE
            || physicalTemporaryBytes == Long.MAX_VALUE) {
            return 0L;
        }
        return Math.max(0L, logicalTemporaryBytesSum - physicalTemporaryBytes);
    }

    /**
     * Deterministic diagnostic dump explaining every lifetime and assignment.
     */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("memory planning:\n");
        result.append("  strategy=linear-scan\n");
        result.append("  logical-temporaries=").append(logicalTemporaryCount).append('\n');
        result.append("  materialized-logical=").append(materializedLogicalTemporaryCount).append('\n');
        result.append("  elided-by-fusion=").append(elidedLogicalTemporaryCount).append('\n');
        result.append("  physical-slots=").append(physicalSlotCount()).append('\n');
        result.append("  logical-temporary-bytes-sum=").append(logicalTemporaryBytesSum).append('\n');
        result.append("  peak-live-temporary-bytes=").append(peakLiveTemporaryBytes).append('\n');
        result.append("  physical-temporary-bytes=").append(physicalTemporaryBytes).append('\n');
        result.append("  logical-to-physical-reuse-count=").append(logicalToPhysicalReuseCount).append('\n');
        result.append("  allocation-count=").append(allocationCount()).append('\n');
        result.append("  peak-concurrent-slots=").append(peakConcurrentSlots).append('\n');
        result.append("  reused-slots=").append(reusedSlotCount).append('\n');
        result.append("  bytes-avoided=").append(bytesAvoided()).append('\n');
        result.append("  lifetimes:\n");
        if (lifetimes.isEmpty()) {
            result.append("    (none)\n");
        } else {
            for (ExecutionLifetime lifetime : lifetimes) {
                LogicalBuffer buffer = lifetime.logicalBuffer();
                result.append("    %").append(buffer.id()).append(' ')
                    .append(buffer.name()).append('\n');
                result.append("      lifetime=").append(lifetime).append('\n');
                result.append("      consumers=")
                    .append(formatConsumers(lifetime)).append('\n');
                PhysicalBufferSlot slot = slotFor(buffer);
                result.append("      slot=").append(slot == null ? "(none)" : slot.displayId());
                if (slot != null && slot.logicalBuffers().indexOf(buffer) > 0) {
                    result.append(" REUSE");
                }
                result.append('\n');
            }
        }
        result.append("  slots:\n");
        if (slots.isEmpty()) {
            result.append("    (none)\n");
        } else {
            for (PhysicalBufferSlot slot : slots) {
                result.append("    slot ").append(slot.displayId()).append('\n');
                result.append("      class=").append(slot.storageClass().key()).append('\n');
                result.append("      shape=").append(slot.shape()).append('\n');
                result.append("      bytes=").append(slot.payloadBytes()).append('\n');
                result.append("      logical:\n");
                for (LogicalBuffer buffer : slot.logicalBuffers()) {
                    ExecutionLifetime lifetime = lifetime(buffer);
                    result.append("        %").append(buffer.id()).append(' ')
                        .append(buffer.name()).append(' ').append(lifetime).append('\n');
                }
            }
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private void validateAssignments() {
        for (Map.Entry<LogicalBuffer, PhysicalBufferSlot> entry : slotsByBuffer.entrySet()) {
            LogicalBuffer buffer = entry.getKey();
            PhysicalBufferSlot slot = entry.getValue();
            if (!slot.logicalBuffers().contains(buffer)) {
                throw new IllegalArgumentException("Slot assignment is not reflected by slot membership");
            }
            for (LogicalBuffer other : slot.logicalBuffers()) {
                if (other != buffer && interferes(buffer, other)) {
                    throw new IllegalArgumentException(
                        "Interfering logical buffers share physical slot P" + slot.slotId());
                }
            }
        }
        for (ExecutionLifetime lifetime : lifetimes) {
            if (!slotsByBuffer.containsKey(lifetime.logicalBuffer())) {
                throw new IllegalArgumentException("Materialized lifetime has no physical slot");
            }
        }
    }

    private static String formatConsumers(ExecutionLifetime lifetime) {
        if (lifetime.firstConsumerStep() == null) {
            return "none";
        }
        return "step" + lifetime.firstConsumerStep() + "..step" + lifetime.lastConsumerStep();
    }

    private static int requireNonNegative(int value, String role) {
        if (value < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
        return value;
    }

    private static long requireNonNegative(long value, String role) {
        if (value < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
        return value;
    }

    private static <T> List<T> immutableCopy(List<T> source, String role) {
        if (source == null || source.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(role + " must not be null or contain nulls");
        }
        return Collections.unmodifiableList(new ArrayList<>(source));
    }
}
