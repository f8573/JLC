package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Immutable selected operation graph produced by {@link MatrixCompiler}.
 *
 * <p>The plan can be inspected without executing it. Evaluation materializes
 * each selected operation as a temporary value and memoizes shared plan nodes
 * by identity; fusion is intentionally outside M1.</p>
 */
public final class ExecutionPlan {
    private final PlanNode root;
    private final OptimizationSemantics semantics;
    private final MatrixCostModel costModel;
    private final CostEstimate estimatedCost;

    public ExecutionPlan(PlanNode root,
                         OptimizationSemantics semantics,
                         MatrixCostModel costModel) {
        PlanNode.requireNode(root, "Execution plan root");
        if (semantics == null) {
            throw new IllegalArgumentException("Optimization semantics must not be null");
        }
        if (costModel == null) {
            throw new IllegalArgumentException("Cost model must not be null");
        }
        this.root = root;
        this.semantics = semantics;
        this.costModel = costModel;
        this.estimatedCost = PlanCosts.uniqueCost(root, costModel);
    }

    public PlanNode root() {
        return root;
    }

    public PlanNode operationTree() {
        return root;
    }

    public MatrixShape outputShape() {
        return root.shape();
    }

    public MatrixShape shape() {
        return outputShape();
    }

    public OptimizationSemantics semantics() {
        return semantics;
    }

    public OptimizationSemantics optimizationSemantics() {
        return semantics;
    }

    public MatrixCostModel costModel() {
        return costModel;
    }

    public CostEstimate estimatedCost() {
        return estimatedCost;
    }

    public CostEstimate cost() {
        return estimatedCost;
    }

    public long scalarMultiplicationCost() {
        return estimatedCost.scalarMultiplications();
    }

    public long estimatedScalarMultiplications() {
        return scalarMultiplicationCost();
    }

    public long estimatedOutputBytes() {
        return estimatedCost.estimatedOutputBytes();
    }

    /**
     * Render the selected operation tree as a deterministic nested form.
     */
    public String expression() {
        return PlanPrinter.expression(root);
    }

    /**
     * Render a deterministic SSA-like dump that makes DAG sharing visible.
     */
    public String dump() {
        return PlanPrinter.dump(root);
    }

    /**
     * Evaluate this plan using the existing JLC matrix operations.
     */
    public Matrix evaluate() {
        return MatrixPlanEvaluator.evaluate(root);
    }

    @Override
    public String toString() {
        return expression();
    }
}
