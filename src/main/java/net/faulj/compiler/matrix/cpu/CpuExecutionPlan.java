package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleAnnotation;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;
import net.faulj.matrix.Matrix;

/**
 * Immutable, inspectable, executable CPU plan produced by M4 and the selected
 * post-M4 fusion lowering.
 *
 * <p>The plan retains logical buffers from M2 and the schedule from M3. R1
 * adds an immutable physical-memory description derived from the final CPU
 * steps; live Matrix objects remain execution-scoped and are never retained
 * by this plan.</p>
 */
public final class CpuExecutionPlan {
    private final SchedulePlan schedule;
    private final List<CpuStep> steps;
    private final List<CpuBufferBinding> inputBindings;
    private final List<LogicalBuffer> temporaryBuffers;
    private final List<LogicalBuffer> materializedTemporaryBuffers;
    private final List<LogicalBuffer> elidedTemporaryBuffers;
    private final LogicalBuffer outputBuffer;
    private final boolean hasParallelAnnotations;
    private final boolean hasVectorAnnotations;
    private final long estimatedTemporaryBytes;
    private final FusionMetrics fusionMetrics;
    private final PhysicalMemoryPlan physicalMemoryPlan;

    CpuExecutionPlan(SchedulePlan schedule,
                     List<CpuStep> steps,
                     List<CpuBufferBinding> inputBindings,
                     List<LogicalBuffer> temporaryBuffers,
                     List<LogicalBuffer> materializedTemporaryBuffers,
                     List<LogicalBuffer> elidedTemporaryBuffers) {
        this(
            schedule,
            steps,
            inputBindings,
            temporaryBuffers,
            materializedTemporaryBuffers,
            elidedTemporaryBuffers,
            FusionMetrics.empty(FusionStrategy.fromSystemProperty(), schedule.program(), steps));
    }

    CpuExecutionPlan(SchedulePlan schedule,
                     List<CpuStep> steps,
                     List<CpuBufferBinding> inputBindings,
                     List<LogicalBuffer> temporaryBuffers,
                     List<LogicalBuffer> materializedTemporaryBuffers,
                     List<LogicalBuffer> elidedTemporaryBuffers,
                     FusionMetrics fusionMetrics) {
        this.schedule = Objects.requireNonNull(schedule, "CPU schedule must not be null");
        this.steps = immutableCopy(steps, "CPU steps");
        this.inputBindings = immutableCopy(inputBindings, "CPU input bindings");
        this.temporaryBuffers = immutableCopy(temporaryBuffers, "CPU temporary buffers");
        this.materializedTemporaryBuffers = immutableCopy(
            materializedTemporaryBuffers, "CPU materialized temporary buffers");
        this.elidedTemporaryBuffers = immutableCopy(
            elidedTemporaryBuffers, "CPU elided temporary buffers");
        this.fusionMetrics = Objects.requireNonNull(
            fusionMetrics, "CPU fusion metrics must not be null");
        this.outputBuffer = schedule.program().resultBuffer();
        validateStepIds();
        validateBuffers();
        this.hasParallelAnnotations = hasAnnotation(ScheduleAnnotation.PARALLEL)
            || hasAnnotation(ScheduleAnnotation.REDUCTION_PARALLEL_ELIGIBLE);
        this.hasVectorAnnotations = hasAnnotation(ScheduleAnnotation.VECTOR);
        this.estimatedTemporaryBytes = estimatedBytes(this.materializedTemporaryBuffers);
        this.physicalMemoryPlan = PhysicalMemoryPlanner.plan(this);
    }

    public SchedulePlan schedule() {
        return schedule;
    }

    public SchedulePlan schedulePlan() {
        return schedule;
    }

    public AffineProgram affineProgram() {
        return schedule.program();
    }

    public OptimizationSemantics semantics() {
        return schedule.program().semantics();
    }

    public List<CpuStep> steps() {
        return steps;
    }

    public List<CpuBufferBinding> inputBindings() {
        return inputBindings;
    }

    public List<LogicalBuffer> temporaryBuffers() {
        return temporaryBuffers;
    }

    /** Logical temporaries that receive an execution-owned Matrix. */
    public List<LogicalBuffer> materializedTemporaryBuffers() {
        return materializedTemporaryBuffers;
    }

    /** Logical temporaries omitted by a realized fusion. */
    public List<LogicalBuffer> elidedTemporaryBuffers() {
        return elidedTemporaryBuffers;
    }

    /** Immutable R1 physical plan; it contains no live runtime allocations. */
    public PhysicalMemoryPlan physicalMemoryPlan() {
        return physicalMemoryPlan;
    }

    /** Short alias for diagnostics and callers that use the memory-plan term. */
    public PhysicalMemoryPlan memoryPlan() {
        return physicalMemoryPlan;
    }

    /** Immutable accounting and planner decisions for the final CPU plan. */
    public FusionMetrics fusionMetrics() {
        return fusionMetrics;
    }

    /** Short alias for callers that use the generic metrics term. */
    public FusionMetrics metrics() {
        return fusionMetrics;
    }

    public FusionStrategy fusionStrategy() {
        return fusionMetrics.strategy();
    }

    public int physicalSlotCount() {
        return physicalMemoryPlan.physicalSlotCount();
    }

    public long logicalTemporaryBytesSum() {
        return physicalMemoryPlan.logicalTemporaryBytesSum();
    }

    public long peakLiveTemporaryBytes() {
        return physicalMemoryPlan.peakLiveTemporaryBytes();
    }

    public long physicalTemporaryBytes() {
        return physicalMemoryPlan.physicalTemporaryBytes();
    }

    public int logicalToPhysicalReuseCount() {
        return physicalMemoryPlan.logicalToPhysicalReuseCount();
    }

    public int allocationCount() {
        return physicalMemoryPlan.allocationCount();
    }

    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public int stepCount() {
        return steps.size();
    }

    public long gemmStepCount() {
        return steps.stream().filter(step -> step.kind() == CpuStepKind.GEMM).count();
    }

    public int temporaryCount() {
        return temporaryBuffers.size();
    }

    /** Number of logical temporary buffers before fusion elision. */
    public int logicalTemporaryCount() {
        return physicalMemoryPlan.logicalTemporaryCount();
    }

    public int materializedTemporaryCount() {
        return materializedTemporaryBuffers.size();
    }

    public int elidedTemporaryCount() {
        return elidedTemporaryBuffers.size();
    }

    public long estimatedTemporaryBytes() {
        return estimatedTemporaryBytes;
    }

    public boolean hasFusedElementwiseStep() {
        return steps.stream().anyMatch(step -> step.kind() == CpuStepKind.FUSED_ELEMENTWISE);
    }

    public boolean hasFusedRegionStep() {
        return steps.stream().anyMatch(step -> step instanceof CpuFusedRegionStep);
    }

    public List<FusedRegionPlan> fusedRegions() {
        return steps.stream()
            .filter(CpuFusedRegionStep.class::isInstance)
            .map(CpuFusedRegionStep.class::cast)
            .map(CpuFusedRegionStep::regionPlan)
            .toList();
    }

    public int fusedRegionCount() {
        return fusionMetrics.fusedRegionCount();
    }

    public int fusionElidedTemporaryCount() {
        return fusionMetrics.fusionElidedTemporaryCount();
    }

    public long estimatedFusionElidedBytes() {
        return fusionMetrics.fusionElidedBytes();
    }

    public boolean hasParallelAnnotations() {
        return hasParallelAnnotations;
    }

    public boolean hasVectorAnnotations() {
        return hasVectorAnnotations;
    }

    /** M4 intentionally preserves these eligibility annotations but executes serially. */
    public boolean realizesParallelAnnotations() {
        return false;
    }

    /** M4 intentionally preserves vector eligibility but uses scalar loops. */
    public boolean realizesVectorAnnotations() {
        return false;
    }

    /** Execute using the borrowed bindings captured from runtime Input nodes. */
    public Matrix execute() {
        return CpuExecutor.execute(this);
    }

    /** Execute with explicit borrowed bindings for a plan-only lowering. */
    public Matrix execute(java.util.Map<LogicalBuffer, Matrix> bindings) {
        return CpuExecutor.execute(this, bindings);
    }

    /**
     * Deterministic human-readable CPU lowering dump.
     */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("cpu execution plan:\n");
        result.append("inputs:\n");
        if (inputBindings.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (CpuBufferBinding binding : inputBindings) {
                result.append("  ").append(binding).append('\n');
            }
        }
        result.append("temporaries:\n");
        result.append("  materialized (execution-owned):\n");
        appendBuffers(result, materializedTemporaryBuffers);
        result.append("  elided by fusion:\n");
        appendBuffers(result, elidedTemporaryBuffers);
        result.append("fusion planning:\n");
        result.append(fusionMetrics.dump().indent(2));
        result.append("  accepted regions:\n");
        List<FusedRegionPlan> regions = fusedRegions();
        if (regions.isEmpty()) {
            result.append("    (none)\n");
        } else {
            for (FusedRegionPlan region : regions) {
                result.append(region.dump().indent(4));
            }
        }
        result.append(physicalMemoryPlan.dump());
        result.append("steps:\n");
        if (steps.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (CpuStep step : steps) {
                result.append("  ").append(step.dump()).append('\n');
            }
        }
        result.append("annotations:\n");
        if (!hasParallelAnnotations && !hasVectorAnnotations) {
            result.append("  (none)\n");
        } else {
            if (hasParallelAnnotations) {
                result.append("  parallel eligibility preserved; realization=serial fallback\n");
            }
            if (hasVectorAnnotations) {
                result.append("  vector eligibility preserved; realization=scalar fallback\n");
            }
        }
        result.append("output:\n  %").append(outputBuffer.id()).append(' ')
            .append(outputBuffer.name()).append('\n');
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private void validateStepIds() {
        for (int index = 0; index < steps.size(); index++) {
            if (steps.get(index).id() != index) {
                throw new IllegalArgumentException("CPU step IDs must be contiguous and deterministic");
            }
        }
    }

    private void validateBuffers() {
        for (LogicalBuffer buffer : temporaryBuffers) {
            if (!buffer.isTemporary()) {
                throw new IllegalArgumentException("Temporary list contains a non-temporary buffer");
            }
        }
        for (LogicalBuffer buffer : materializedTemporaryBuffers) {
            if (!temporaryBuffers.contains(buffer)) {
                throw new IllegalArgumentException("Materialized buffer is not a program temporary");
            }
        }
        for (LogicalBuffer buffer : elidedTemporaryBuffers) {
            if (!temporaryBuffers.contains(buffer)) {
                throw new IllegalArgumentException("Elided buffer is not a program temporary");
            }
        }
        if (materializedTemporaryBuffers.stream().anyMatch(elidedTemporaryBuffers::contains)) {
            throw new IllegalArgumentException("A temporary cannot be both materialized and elided");
        }
    }

    private boolean hasAnnotation(ScheduleAnnotation annotation) {
        for (ScheduleRegion region : schedule.regions()) {
            if (region.band().loops().stream().anyMatch(loop -> loop.hasAnnotation(annotation))) {
                return true;
            }
        }
        return false;
    }

    private static void appendBuffers(StringBuilder result, List<LogicalBuffer> buffers) {
        if (buffers.isEmpty()) {
            result.append("    (none)\n");
            return;
        }
        for (LogicalBuffer buffer : buffers) {
            result.append("    %").append(buffer.id()).append(' ')
                .append(buffer.name()).append(' ').append(buffer.shape()).append('\n');
        }
    }

    private static long estimatedBytes(List<LogicalBuffer> buffers) {
        long result = 0L;
        for (LogicalBuffer buffer : buffers) {
            long elements = (long) buffer.shape().rows() * buffer.shape().columns();
            long bytes;
            try {
                bytes = Math.multiplyExact(elements, Double.BYTES);
                result = Math.addExact(result, bytes);
            } catch (ArithmeticException overflow) {
                return Long.MAX_VALUE;
            }
        }
        return result;
    }

    private static <T> List<T> immutableCopy(List<T> source, String role) {
        if (source == null) {
            throw new IllegalArgumentException(role + " must not be null");
        }
        ArrayList<T> copy = new ArrayList<>(source);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(role + " must not contain nulls");
        }
        return Collections.unmodifiableList(copy);
    }
}
