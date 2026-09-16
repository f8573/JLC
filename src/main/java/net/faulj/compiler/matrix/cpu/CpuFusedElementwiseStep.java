package net.faulj.compiler.matrix.cpu;

import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.compiler.matrix.schedule.ScheduleSequence;
import net.faulj.compiler.matrix.schedule.ScheduleStatement;
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
    private final boolean canonicalIdentity;
    private volatile ExecutionPath lastExecutionPath;

    enum ExecutionPath { FAST_REAL_IDENTITY, GENERIC_SCHEDULE }

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
        this.canonicalIdentity = isCanonicalIdentity();
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

    ExecutionPath lastExecutionPath() {
        return lastExecutionPath;
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
        boolean scaledComplex = scaled.hasImagData();
        boolean addendComplex = addend.hasImagData();
        boolean complex = scaledComplex || addendComplex;
        if (canUseFastPath(output, scaled, addend)) {
            double[] out = output.getRawData();
            double[] a = scaled.getRawData();
            double[] b = addend.getRawData();
            int count = out.length;
            if (scaledOperandFirst) {
                for (int p = 0; p < count; p++) {
                    out[p] = factor * a[p] + b[p];
                }
            } else {
                for (int p = 0; p < count; p++) {
                    out[p] = b[p] + factor * a[p];
                }
            }
            lastExecutionPath = ExecutionPath.FAST_REAL_IDENTITY;
            return;
        }
        lastExecutionPath = ExecutionPath.GENERIC_SCHEDULE;
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
                double imaginary;
                if (scaledComplex && addendComplex) {
                    double scaledImaginary = factor
                        * scaled.getImag(scaleIndices[0], scaleIndices[1]);
                    double addImaginary = addend.getImag(addIndices[0], addIndices[1]);
                    imaginary = scaledOperandFirst
                        ? scaledImaginary + addImaginary : addImaginary + scaledImaginary;
                } else if (scaledComplex) {
                    // Matrix.add copies a lone imaginary array; it does not add +0.0.
                    imaginary = factor * scaled.getImag(scaleIndices[0], scaleIndices[1]);
                } else {
                    // Eager scale of a real matrix has no imaginary lane, even for NaN/Inf.
                    imaginary = addend.getImag(addIndices[0], addIndices[1]);
                }
                output.setComplex(outputIndices[0], outputIndices[1], real, imaginary);
            } else {
                output.set(outputIndices[0], outputIndices[1], real);
            }
        });
    }

    private boolean canUseFastPath(Matrix output, Matrix scaled, Matrix addend) {
        if (!canonicalIdentity || output.getClass() != Matrix.class
            || scaled.getClass() != Matrix.class || addend.getClass() != Matrix.class
            || output.hasImagData() || scaled.hasImagData() || addend.hasImagData()
            || output == scaled || output == addend
            || outputBuffer.ownership() != BufferOwnership.OWNED) {
            return false;
        }
        int rows = output.getRowCount();
        int columns = output.getColumnCount();
        if (scaled.getRowCount() != rows || addend.getRowCount() != rows
            || scaled.getColumnCount() != columns || addend.getColumnCount() != columns) {
            return false;
        }
        long count = (long) rows * columns;
        return count == output.getRawData().length
            && count == scaled.getRawData().length
            && count == addend.getRawData().length;
    }

    private boolean isCanonicalIdentity() {
        if (scheduleBand.loops().size() != 2
            || !(scheduleBand.body() instanceof ScheduleSequence sequence)
            || sequence.children().size() != 2
            || !(sequence.children().get(0) instanceof ScheduleStatement scaleStatement)
            || !(sequence.children().get(1) instanceof ScheduleStatement addStatement)
            || scaleStatement.id() != scaleStatementId || addStatement.id() != addStatementId
            || outputAccess.buffer() != outputBuffer
            || scaleAccess.buffer() != scaledOperand || addAccess.buffer() != addOperand
            || outputAccess.indices().size() != 2 || scaleAccess.indices().size() != 2
            || addAccess.indices().size() != 2) {
            return false;
        }
        ScheduleLoop row = scheduleBand.loop(0);
        ScheduleLoop column = scheduleBand.loop(1);
        if (!isIdentityLoop(row, outputBuffer.shape().rows())
            || !isIdentityLoop(column, outputBuffer.shape().columns())) {
            return false;
        }
        AffineExpr rowIndex = AffineExpr.variable(row.inductionVariable());
        AffineExpr columnIndex = AffineExpr.variable(column.inductionVariable());
        return hasIdentityIndices(outputAccess, rowIndex, columnIndex)
            && hasIdentityIndices(scaleAccess, rowIndex, columnIndex)
            && hasIdentityIndices(addAccess, rowIndex, columnIndex);
    }

    private static boolean isIdentityLoop(ScheduleLoop loop, int extent) {
        return loop.inductionVariable().equals(loop.semanticVariable())
            && loop.lowerBound() == 0 && loop.upperBound() == extent && loop.step() == 1
            && loop.annotations().isEmpty() && loop.guard() == null
            && AffineExpr.variable(loop.inductionVariable()).equals(loop.valueExpression());
    }

    private static boolean hasIdentityIndices(AffineAccess access,
                                               AffineExpr row, AffineExpr column) {
        return access.index(0).equals(row) && access.index(1).equals(column);
    }
}
