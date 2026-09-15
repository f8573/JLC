package net.faulj.compiler.matrix.cpu;

import java.util.List;

import net.faulj.compiler.matrix.affine.LogicalBuffer;

/**
 * One immutable operation in a {@link CpuExecutionPlan}.
 *
 * <p>The interface is intentionally descriptive. Execution is performed by
 * {@link CpuExecutor}, so a CPU step cannot become an independent expression
 * interpreter or allocator.</p>
 */
public interface CpuStep {
    int id();

    CpuStepKind kind();

    LogicalBuffer outputBuffer();

    List<LogicalBuffer> inputBuffers();

    /** Deterministic one-line description used by plan dumps. */
    String description();

    default String dump() {
        return "P" + id() + " " + description();
    }
}
