package net.faulj.compiler.matrix.cpu;

import java.util.Map;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;

/** Executes an already lowered immutable {@link CpuExecutionPlan}. */
public final class CpuExecutor {
    private CpuExecutor() {
    }

    public static Matrix execute(CpuExecutionPlan plan) {
        return execute(plan, Map.of());
    }

    public static Matrix execute(CpuExecutionPlan plan,
                                 Map<LogicalBuffer, Matrix> runtimeBindings) {
        if (plan == null) {
            throw new IllegalArgumentException("CPU execution plan must not be null");
        }
        if (runtimeBindings == null) {
            throw new IllegalArgumentException("CPU runtime bindings must not be null");
        }

        for (LogicalBuffer buffer : plan.affineProgram().buffers()) {
            if (buffer.isSymbolic()) {
                throw new IllegalStateException(
                    "Cannot execute CPU plan: symbolic input buffer %" + buffer.id()
                        + " has no runtime storage");
            }
        }

        CpuExecutionContext context = new CpuExecutionContext();
        for (CpuBufferBinding captured : plan.inputBindings()) {
            Matrix matrix = captured.matrix();
            Matrix supplied = runtimeBindings.get(captured.buffer());
            if (supplied != null) {
                matrix = supplied;
            }
            if (matrix == null) {
                throw new IllegalStateException(
                    "Cannot execute CPU plan: external input buffer %"
                        + captured.buffer().id() + " is unbound");
            }
            validateShape(captured.buffer(), matrix);
            context.bind(captured.buffer(), matrix);
        }

        Matrix result = null;
        Throwable failure = null;
        try {
            for (CpuStep step : plan.steps()) {
                if (step instanceof CpuGemmStep gemm) {
                    gemm.execute(context);
                } else if (step instanceof CpuElementwiseStep elementwise) {
                    elementwise.execute(context);
                } else if (step instanceof CpuTransposeStep transpose) {
                    transpose.execute(context);
                } else if (step instanceof CpuFusedElementwiseStep fused) {
                    fused.execute(context);
                } else {
                    throw new IllegalStateException(
                        "Unsupported CPU step type: " + step.getClass().getName());
                }
            }
            result = context.value(plan.outputBuffer());
            return result;
        } catch (RuntimeException | Error exception) {
            failure = exception;
            throw exception;
        } finally {
            context.closeOwnedExcept(result, failure);
        }
    }

    private static void validateShape(LogicalBuffer buffer, Matrix matrix) {
        if (matrix.getRowCount() != buffer.shape().rows()
            || matrix.getColumnCount() != buffer.shape().columns()) {
            throw new IllegalArgumentException(
                "Runtime matrix shape " + matrix.getRowCount() + "x" + matrix.getColumnCount()
                    + " does not match logical buffer %" + buffer.id()
                    + " shape " + buffer.shape());
        }
    }
}
