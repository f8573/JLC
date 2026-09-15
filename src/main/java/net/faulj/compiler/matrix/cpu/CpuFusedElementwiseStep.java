package net.faulj.compiler.matrix.cpu;

import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.matrix.Matrix;

/**
 * One-loop realization of the narrow M4 fusion family
 * {@code add(scale(A, alpha), B)}.
 */
public final class CpuFusedElementwiseStep implements CpuStep {
    private final int id;
    private final int scaleStatementId;
    private final int addStatementId;
    private final ScheduleBand scheduleBand;
    private final LogicalBuffer outputBuffer;
    private final LogicalBuffer scaledOperand;
    private final LogicalBuffer addOperand;
    private final LogicalBuffer eliminatedBuffer;
    private final double factor;
    private final boolean scaledOperandFirst;
    private final AffineAccess outputAccess;
    private final AffineAccess scaleAccess;
    private final AffineAccess addAccess;

    CpuFusedElementwiseStep(int id,
                            int scaleStatementId,
                            int addStatementId,
                            ScheduleBand scheduleBand,
                            LogicalBuffer outputBuffer,
                            LogicalBuffer scaledOperand,
                            LogicalBuffer addOperand,
                            LogicalBuffer eliminatedBuffer,
                            double factor,
                            boolean scaledOperandFirst,
                            AffineAccess outputAccess,
                            AffineAccess scaleAccess,
                            AffineAccess addAccess) {
        if (id < 0 || scaleStatementId < 0 || addStatementId < 0) {
            throw new IllegalArgumentException("CPU and statement IDs must be non-negative");
        }
        this.id = id;
        this.scaleStatementId = scaleStatementId;
        this.addStatementId = addStatementId;
        this.scheduleBand = Objects.requireNonNull(scheduleBand, "Fused schedule must not be null");
        this.outputBuffer = Objects.requireNonNull(outputBuffer, "Fused output buffer must not be null");
        this.scaledOperand = Objects.requireNonNull(scaledOperand, "Fused scale operand must not be null");
        this.addOperand = Objects.requireNonNull(addOperand, "Fused add operand must not be null");
        this.eliminatedBuffer = Objects.requireNonNull(
            eliminatedBuffer, "Fused eliminated buffer must not be null");
        this.factor = factor;
        this.scaledOperandFirst = scaledOperandFirst;
        this.outputAccess = Objects.requireNonNull(outputAccess, "Fused output access must not be null");
        this.scaleAccess = Objects.requireNonNull(scaleAccess, "Fused scale access must not be null");
        this.addAccess = Objects.requireNonNull(addAccess, "Fused add access must not be null");
    }

    @Override
    public int id() {
        return id;
    }

    public int scaleStatementId() {
        return scaleStatementId;
    }

    public int addStatementId() {
        return addStatementId;
    }

    public ScheduleBand scheduleBand() {
        return scheduleBand;
    }

    @Override
    public CpuStepKind kind() {
        return CpuStepKind.FUSED_ELEMENTWISE;
    }

    @Override
    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public LogicalBuffer scaledOperand() {
        return scaledOperand;
    }

    public LogicalBuffer addOperand() {
        return addOperand;
    }

    /** The scale result that is intentionally not materialized by this step. */
    public LogicalBuffer eliminatedBuffer() {
        return eliminatedBuffer;
    }

    public double factor() {
        return factor;
    }

    public boolean scaledOperandFirst() {
        return scaledOperandFirst;
    }

    public AffineAccess outputAccess() {
        return outputAccess;
    }

    public AffineAccess scaleAccess() {
        return scaleAccess;
    }

    public AffineAccess addAccess() {
        return addAccess;
    }

    @Override
    public List<LogicalBuffer> inputBuffers() {
        return List.of(scaledOperand, addOperand);
    }

    @Override
    public String description() {
        return "FUSED_ELEMENTWISE " + outputAccess.location() + " = "
            + (scaledOperandFirst
                ? Double.toString(factor) + " * " + scaleAccess.location()
                    + " + " + addAccess.location()
                : addAccess.location() + " + " + Double.toString(factor)
                    + " * " + scaleAccess.location())
            + " (elides %" + eliminatedBuffer.id() + ")";
    }

    void execute(CpuExecutionContext context) {
        Matrix output = context.allocate(outputBuffer);
        Matrix scaled = context.value(scaledOperand);
        Matrix addend = context.value(addOperand);
        boolean complex = scaled.hasImagData() || addend.hasImagData();
        if (complex) {
            output.ensureImagData();
        }

        CpuLoopExecutor.forEachPoint(scheduleBand, bindings -> {
            int[] outputIndices = CpuLoopExecutor.indices(outputAccess, bindings);
            int[] scaleIndices = CpuLoopExecutor.indices(scaleAccess, bindings);
            int[] addIndices = CpuLoopExecutor.indices(addAccess, bindings);
            if (!CpuLoopExecutor.inBounds(outputAccess, output, outputIndices)
                || !CpuLoopExecutor.inBounds(scaleAccess, scaled, scaleIndices)
                || !CpuLoopExecutor.inBounds(addAccess, addend, addIndices)) {
                return;
            }
            double scaledReal = factor * scaled.get(scaleIndices[0], scaleIndices[1]);
            double addReal = addend.get(addIndices[0], addIndices[1]);
            double real = scaledOperandFirst ? scaledReal + addReal : addReal + scaledReal;
            if (complex) {
                double scaledImaginary = factor * scaled.getImag(scaleIndices[0], scaleIndices[1]);
                double addImaginary = addend.getImag(addIndices[0], addIndices[1]);
                double imaginary = scaledOperandFirst
                    ? scaledImaginary + addImaginary : addImaginary + scaledImaginary;
                output.setComplex(outputIndices[0], outputIndices[1], real, imaginary);
            } else {
                output.set(outputIndices[0], outputIndices[1], real);
            }
        });
    }
}
