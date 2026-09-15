package net.faulj.compiler.matrix.cpu;

import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;
import net.faulj.kernels.gemm.Gemm;

/** Opaque production GEMM invocation in a CPU execution plan. */
public final class CpuGemmStep implements CpuStep {
    private final int id;
    private final LogicalBuffer outputBuffer;
    private final LogicalBuffer lhs;
    private final LogicalBuffer rhs;
    private final int initStatementId;
    private final int updateStatementId;

    CpuGemmStep(int id,
                LogicalBuffer outputBuffer,
                LogicalBuffer lhs,
                LogicalBuffer rhs,
                int initStatementId,
                int updateStatementId) {
        if (id < 0) {
            throw new IllegalArgumentException("CPU step ID must be non-negative");
        }
        this.id = id;
        this.outputBuffer = Objects.requireNonNull(outputBuffer, "GEMM output buffer must not be null");
        this.lhs = Objects.requireNonNull(lhs, "GEMM left buffer must not be null");
        this.rhs = Objects.requireNonNull(rhs, "GEMM right buffer must not be null");
        this.initStatementId = requireStatementId(initStatementId);
        this.updateStatementId = requireStatementId(updateStatementId);
    }

    @Override
    public int id() {
        return id;
    }

    @Override
    public CpuStepKind kind() {
        return CpuStepKind.GEMM;
    }

    @Override
    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public LogicalBuffer lhs() {
        return lhs;
    }

    public LogicalBuffer rhs() {
        return rhs;
    }

    public int initStatementId() {
        return initStatementId;
    }

    public int updateStatementId() {
        return updateStatementId;
    }

    @Override
    public List<LogicalBuffer> inputBuffers() {
        return List.of(lhs, rhs);
    }

    @Override
    public String description() {
        return "GEMM %" + outputBuffer.id() + " = %" + lhs.id() + " * %" + rhs.id();
    }

    void execute(CpuExecutionContext context) {
        Matrix left = context.value(lhs);
        Matrix right = context.value(rhs);
        context.bindOwned(outputBuffer, Gemm.multiply(left, right));
    }

    private static int requireStatementId(int statementId) {
        if (statementId < 0) {
            throw new IllegalArgumentException("GEMM statement ID must be non-negative");
        }
        return statementId;
    }
}
