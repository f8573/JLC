package net.faulj.compiler.matrix.cpu;

import java.util.Objects;

import net.faulj.compiler.matrix.affine.LogicalBuffer;

/**
 * Immutable executable lifetime for one materialized logical buffer.
 *
 * <p>The producer and consumer positions are CPU-step IDs, not M2 statement
 * IDs. The interval exposed by {@link #startStep()} and {@link #endStep()} is
 * inclusive, so touching the same step is conservatively an interference.</p>
 */
public final class ExecutionLifetime {
    private final LogicalBuffer logicalBuffer;
    private final int producingStep;
    private final Integer firstConsumerStep;
    private final Integer lastConsumerStep;
    private final boolean liveThroughReturn;
    private final int endStep;

    public ExecutionLifetime(LogicalBuffer logicalBuffer,
                             int producingStep,
                             Integer firstConsumerStep,
                             Integer lastConsumerStep,
                             boolean liveThroughReturn,
                             int endStep) {
        this.logicalBuffer = Objects.requireNonNull(
            logicalBuffer, "Execution lifetime buffer must not be null");
        if (producingStep < 0 || endStep < producingStep) {
            throw new IllegalArgumentException("Execution lifetime step range is invalid");
        }
        validateConsumer(firstConsumerStep, "first consumer");
        validateConsumer(lastConsumerStep, "last consumer");
        if (firstConsumerStep != null && lastConsumerStep != null
            && firstConsumerStep > lastConsumerStep) {
            throw new IllegalArgumentException("First consumer must not follow last consumer");
        }
        if (firstConsumerStep != null && firstConsumerStep < producingStep) {
            throw new IllegalArgumentException("Consumer cannot precede the producer");
        }
        if (lastConsumerStep != null && lastConsumerStep < producingStep) {
            throw new IllegalArgumentException("Consumer cannot precede the producer");
        }
        if (liveThroughReturn && lastConsumerStep != null && endStep < lastConsumerStep) {
            throw new IllegalArgumentException("Return lifetime must include its final consumer");
        }
        this.producingStep = producingStep;
        this.firstConsumerStep = firstConsumerStep;
        this.lastConsumerStep = lastConsumerStep;
        this.liveThroughReturn = liveThroughReturn;
        this.endStep = endStep;
    }

    public LogicalBuffer logicalBuffer() {
        return logicalBuffer;
    }

    public LogicalBuffer buffer() {
        return logicalBuffer;
    }

    public int producingStep() {
        return producingStep;
    }

    public int producerStep() {
        return producingStep;
    }

    public int startStep() {
        return producingStep;
    }

    public Integer firstConsumerStep() {
        return firstConsumerStep;
    }

    public Integer firstConsumer() {
        return firstConsumerStep;
    }

    public Integer lastConsumerStep() {
        return lastConsumerStep;
    }

    public Integer lastConsumer() {
        return lastConsumerStep;
    }

    /** Inclusive final step used by the interference allocator. */
    public int endStep() {
        return endStep;
    }

    public int lastUseStep() {
        return endStep;
    }

    public boolean liveThroughReturn() {
        return liveThroughReturn;
    }

    public boolean containsStep(int step) {
        return step >= startStep() && step <= endStep;
    }

    @Override
    public String toString() {
        return "step" + startStep() + ".." + endStep
            + (liveThroughReturn ? " (live-through-return)" : "");
    }

    private static void validateConsumer(Integer step, String role) {
        if (step != null && step < 0) {
            throw new IllegalArgumentException(role + " step must be non-negative");
        }
    }
}
