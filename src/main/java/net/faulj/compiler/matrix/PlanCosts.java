package net.faulj.compiler.matrix;

import java.util.IdentityHashMap;

/**
 * Computes a plan cost once per plan-node identity, matching DAG execution
 * semantics rather than charging shared producers once per consumer edge.
 */
final class PlanCosts {
    private PlanCosts() {
    }

    static CostEstimate uniqueCost(PlanNode root, MatrixCostModel model) {
        if (root == null || model == null) {
            throw new IllegalArgumentException("Plan root and cost model must not be null");
        }
        return visit(root, model, new IdentityHashMap<>());
    }

    private static CostEstimate visit(PlanNode node,
                                      MatrixCostModel model,
                                      IdentityHashMap<PlanNode, Boolean> seen) {
        if (seen.put(node, Boolean.TRUE) != null) {
            return CostEstimate.ZERO;
        }

        CostEstimate children;
        CostEstimate operation;
        if (node instanceof PlanInput input) {
            children = CostEstimate.ZERO;
            operation = model.estimateInput(input.shape());
        } else if (node instanceof PlanMatMul matMul) {
            children = visit(matMul.lhs(), model, seen)
                .plus(visit(matMul.rhs(), model, seen));
            operation = model.estimateMatMul(matMul.lhs().shape(), matMul.rhs().shape());
        } else if (node instanceof PlanAdd add) {
            children = visit(add.lhs(), model, seen)
                .plus(visit(add.rhs(), model, seen));
            operation = model.estimateAdd(add.shape());
        } else if (node instanceof PlanScale scale) {
            children = visit(scale.operand(), model, seen);
            operation = model.estimateScale(scale.shape());
        } else {
            PlanTranspose transpose = (PlanTranspose) node;
            children = visit(transpose.operand(), model, seen);
            operation = model.estimateTranspose(transpose.shape());
        }
        if (operation == null) {
            throw new IllegalArgumentException("Cost model returned null for "
                + node.getClass().getSimpleName());
        }
        return children.plus(operation);
    }
}
