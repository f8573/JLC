package net.faulj.compiler.matrix;

import java.util.IdentityHashMap;

import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;

/**
 * M1 interpreter for execution plans.
 */
final class MatrixPlanEvaluator {
    private MatrixPlanEvaluator() {
    }

    static Matrix evaluate(PlanNode root) {
        PlanNode.requireNode(root, "Plan root");
        return evaluate(root, new IdentityHashMap<>());
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
}
