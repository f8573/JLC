package net.faulj.compiler.matrix.cpu;

import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.matrix.Matrix;

/** Explicit scalar-loop lowering for Add and Scale. */
public final class CpuElementwiseStep implements CpuStep {
    private final int id;
    private final int statementId;
    private final CpuElementwiseOperation operation;
    private final ScheduleBand scheduleBand;
    private final LogicalBuffer outputBuffer;
    private final LogicalBuffer operand;
    private final LogicalBuffer secondOperand;
    private final double factor;
    private final AffineAccess outputAccess;
    private final AffineAccess operandAccess;
    private final AffineAccess secondOperandAccess;

    private CpuElementwiseStep(int id,
                               int statementId,
                               CpuElementwiseOperation operation,
                               ScheduleBand scheduleBand,
                               LogicalBuffer outputBuffer,
                               LogicalBuffer operand,
                               LogicalBuffer secondOperand,
                               double factor,
                               AffineAccess outputAccess,
                               AffineAccess operandAccess,
                               AffineAccess secondOperandAccess) {
        if (id < 0 || statementId < 0) {
            throw new IllegalArgumentException("CPU and statement IDs must be non-negative");
        }
        this.id = id;
        this.statementId = statementId;
        this.operation = Objects.requireNonNull(operation, "Elementwise operation must not be null");
        this.scheduleBand = Objects.requireNonNull(scheduleBand, "Elementwise schedule must not be null");
        this.outputBuffer = Objects.requireNonNull(outputBuffer, "Elementwise output buffer must not be null");
        this.operand = Objects.requireNonNull(operand, "Elementwise operand must not be null");
        if (operation == CpuElementwiseOperation.ADD) {
            this.secondOperand = Objects.requireNonNull(
                secondOperand, "Add second operand must not be null");
            this.secondOperandAccess = Objects.requireNonNull(
                secondOperandAccess, "Add second access must not be null");
        } else {
            this.secondOperand = null;
            this.secondOperandAccess = null;
        }
        this.factor = factor;
        this.outputAccess = Objects.requireNonNull(outputAccess, "Elementwise output access must not be null");
        this.operandAccess = Objects.requireNonNull(operandAccess, "Elementwise operand access must not be null");
    }

    static CpuElementwiseStep add(int id,
                                  int statementId,
                                  ScheduleBand scheduleBand,
                                  LogicalBuffer outputBuffer,
                                  LogicalBuffer lhs,
                                  LogicalBuffer rhs,
                                  AffineAccess outputAccess,
                                  AffineAccess lhsAccess,
                                  AffineAccess rhsAccess) {
        return new CpuElementwiseStep(
            id, statementId, CpuElementwiseOperation.ADD, scheduleBand,
            outputBuffer, lhs, rhs, Double.NaN,
            outputAccess, lhsAccess, rhsAccess);
    }

    static CpuElementwiseStep scale(int id,
                                    int statementId,
                                    ScheduleBand scheduleBand,
                                    LogicalBuffer outputBuffer,
                                    LogicalBuffer operand,
                                    double factor,
                                    AffineAccess outputAccess,
                                    AffineAccess operandAccess) {
        return new CpuElementwiseStep(
            id, statementId, CpuElementwiseOperation.SCALE, scheduleBand,
            outputBuffer, operand, null, factor,
            outputAccess, operandAccess, null);
    }

    @Override
    public int id() {
        return id;
    }

    public int statementId() {
        return statementId;
    }

    public CpuElementwiseOperation operation() {
        return operation;
    }

    public ScheduleBand scheduleBand() {
        return scheduleBand;
    }

    @Override
    public CpuStepKind kind() {
        return CpuStepKind.ELEMENTWISE;
    }

    @Override
    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public LogicalBuffer operand() {
        return operand;
    }

    public LogicalBuffer secondOperand() {
        return secondOperand;
    }

    public double factor() {
        return factor;
    }

    public AffineAccess outputAccess() {
        return outputAccess;
    }

    public AffineAccess operandAccess() {
        return operandAccess;
    }

    public AffineAccess secondOperandAccess() {
        return secondOperandAccess;
    }

    @Override
    public List<LogicalBuffer> inputBuffers() {
        return operation == CpuElementwiseOperation.ADD
            ? List.of(operand, secondOperand)
            : List.of(operand);
    }

    @Override
    public String description() {
        if (operation == CpuElementwiseOperation.ADD) {
            return "ELEMENTWISE ADD " + outputAccess.location() + " = "
                + operandAccess.location() + " + " + secondOperandAccess.location();
        }
        return "ELEMENTWISE SCALE " + outputAccess.location() + " = "
            + Double.toString(factor) + " * " + operandAccess.location();
    }

    void execute(CpuExecutionContext context) {
        Matrix output = context.allocate(outputBuffer);
        Matrix first = context.value(operand);
        Matrix second = operation == CpuElementwiseOperation.ADD
            ? context.value(secondOperand) : null;
        boolean complex = first.hasImagData()
            || (second != null && second.hasImagData());
        boolean firstComplex = first.hasImagData();
        boolean secondComplex = second != null && second.hasImagData();
        if (complex) {
            output.ensureImagData();
        }

        CpuLoopExecutor.forEachPoint(scheduleBand, bindings -> {
            int[] outputIndices = CpuLoopExecutor.indices(outputAccess, bindings);
            int[] firstIndices = CpuLoopExecutor.indices(operandAccess, bindings);
            if (!CpuLoopExecutor.inBounds(outputAccess, output, outputIndices)
                || !CpuLoopExecutor.inBounds(operandAccess, first, firstIndices)) {
                return;
            }
            double real = first.get(firstIndices[0], firstIndices[1]);
            double imaginary = first.getImag(firstIndices[0], firstIndices[1]);
            if (operation == CpuElementwiseOperation.SCALE) {
                real *= factor;
                imaginary *= factor;
            } else {
                int[] secondIndices = CpuLoopExecutor.indices(secondOperandAccess, bindings);
                if (!CpuLoopExecutor.inBounds(secondOperandAccess, second, secondIndices)) {
                    return;
                }
                real += second.get(secondIndices[0], secondIndices[1]);
                if (firstComplex && secondComplex) {
                    imaginary += second.getImag(secondIndices[0], secondIndices[1]);
                } else if (secondComplex) {
                    imaginary = second.getImag(secondIndices[0], secondIndices[1]);
                }
            }
            if (complex) {
                output.setComplex(outputIndices[0], outputIndices[1], real, imaginary);
            } else {
                output.set(outputIndices[0], outputIndices[1], real);
            }
        });
    }
}
