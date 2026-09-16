package net.faulj.compiler.matrix.cpu;

import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.matrix.Matrix;

/** Explicit CPU loop lowering for matrix transpose. */
public final class CpuTransposeStep implements CpuStep {
    private final int id;
    private final int statementId;
    private final ScheduleBand scheduleBand;
    private final LogicalBuffer outputBuffer;
    private final LogicalBuffer operand;
    private final AffineAccess outputAccess;
    private final AffineAccess operandAccess;

    CpuTransposeStep(int id,
                     int statementId,
                     ScheduleBand scheduleBand,
                     LogicalBuffer outputBuffer,
                     LogicalBuffer operand,
                     AffineAccess outputAccess,
                     AffineAccess operandAccess) {
        if (id < 0 || statementId < 0) {
            throw new IllegalArgumentException("CPU and statement IDs must be non-negative");
        }
        this.id = id;
        this.statementId = statementId;
        this.scheduleBand = Objects.requireNonNull(scheduleBand, "Transpose schedule must not be null");
        this.outputBuffer = Objects.requireNonNull(outputBuffer, "Transpose output buffer must not be null");
        this.operand = Objects.requireNonNull(operand, "Transpose operand must not be null");
        this.outputAccess = Objects.requireNonNull(outputAccess, "Transpose output access must not be null");
        this.operandAccess = Objects.requireNonNull(operandAccess, "Transpose operand access must not be null");
    }

    @Override
    public int id() {
        return id;
    }

    public int statementId() {
        return statementId;
    }

    public ScheduleBand scheduleBand() {
        return scheduleBand;
    }

    @Override
    public CpuStepKind kind() {
        return CpuStepKind.TRANSPOSE;
    }

    @Override
    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public LogicalBuffer operand() {
        return operand;
    }

    public AffineAccess outputAccess() {
        return outputAccess;
    }

    public AffineAccess operandAccess() {
        return operandAccess;
    }

    @Override
    public List<LogicalBuffer> inputBuffers() {
        return List.of(operand);
    }

    @Override
    public String description() {
        return "TRANSPOSE " + outputAccess.location() + " = " + operandAccess.location();
    }

    void execute(CpuExecutionContext context) {
        Matrix output = context.allocate(outputBuffer);
        Matrix input = context.value(operand);
        boolean complex = input.hasImagData();
        if (complex) {
            output.ensureImagData();
        }

        CpuLoopExecutor.forEachPoint(scheduleBand, bindings -> {
            int[] outputIndices = CpuLoopExecutor.indices(outputAccess, bindings);
            int[] inputIndices = CpuLoopExecutor.indices(operandAccess, bindings);
            if (!CpuLoopExecutor.inBounds(outputAccess, output, outputIndices)
                || !CpuLoopExecutor.inBounds(operandAccess, input, inputIndices)) {
                return;
            }
            double real = input.get(inputIndices[0], inputIndices[1]);
            if (complex) {
                output.setComplex(
                    outputIndices[0], outputIndices[1], real,
                    input.getImag(inputIndices[0], inputIndices[1]));
            } else {
                output.set(outputIndices[0], outputIndices[1], real);
            }
        });
    }
}
