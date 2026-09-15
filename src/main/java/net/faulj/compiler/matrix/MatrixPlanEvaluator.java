package net.faulj.compiler.matrix;

import java.util.IdentityHashMap;

import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * M1 interpreter for execution plans.
 */
final class MatrixPlanEvaluator {
    private MatrixPlanEvaluator() {
    }

    static Matrix evaluate(PlanNode root) {
        PlanNode.requireNode(root, "Plan root");
        IdentityHashMap<PlanNode, Matrix> values = new IdentityHashMap<>();
        Matrix result = null;
        Throwable failure = null;
        try {
            result = evaluate(root, values);
            return result;
        } catch (RuntimeException | Error exception) {
            failure = exception;
            throw exception;
        } finally {
            closeOwnedIntermediates(values, result, failure);
        }
    }

    private static Matrix evaluate(PlanNode node,
                                   IdentityHashMap<PlanNode, Matrix> values) {
        Matrix cached = values.get(node);
        if (cached != null) {
            return cached;
        }

        Matrix result;
        if (node instanceof PlanInput input) {
            if (!input.isExecutable()) {
                throw new IllegalStateException(
                    "Cannot evaluate a plan containing symbolic input '"
                        + (input.name() == null ? "<unnamed>" : input.name()) + "'");
            }
            result = input.matrix();
        } else if (node instanceof PlanMatMul matMul) {
            result = Gemm.multiply(
                evaluate(matMul.lhs(), values),
                evaluate(matMul.rhs(), values));
        } else if (node instanceof PlanAdd add) {
            result = evaluate(add.lhs(), values).add(evaluate(add.rhs(), values));
        } else if (node instanceof PlanScale scale) {
            result = evaluate(scale.operand(), values).multiplyScalar(scale.factor());
        } else {
            PlanTranspose transpose = (PlanTranspose) node;
            result = evaluate(transpose.operand(), values).transpose();
        }
        values.put(node, result);
        return result;
    }

    private static void closeOwnedIntermediates(IdentityHashMap<PlanNode, Matrix> values,
                                                Matrix transferredResult,
                                                Throwable failure) {
        IdentityHashMap<Matrix, Boolean> closed = new IdentityHashMap<>();
        RuntimeException cleanupFailure = null;
        for (java.util.Map.Entry<PlanNode, Matrix> entry : values.entrySet()) {
            Matrix matrix = entry.getValue();
            if (entry.getKey() instanceof PlanInput || matrix == transferredResult
                || closed.put(matrix, Boolean.TRUE) != null
                || !(matrix instanceof OffHeapMatrix offHeap)) {
                continue;
            }
            try {
                offHeap.close();
            } catch (RuntimeException exception) {
                if (cleanupFailure == null) {
                    cleanupFailure = exception;
                } else {
                    cleanupFailure.addSuppressed(exception);
                }
            }
        }
        if (cleanupFailure != null) {
            if (failure != null) {
                failure.addSuppressed(cleanupFailure);
            } else {
                throw cleanupFailure;
            }
        }
    }
}
