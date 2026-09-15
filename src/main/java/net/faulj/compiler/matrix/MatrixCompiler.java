package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Entry point for the opt-in matrix-expression compiler.
 *
 * <p>The default is {@link OptimizationSemantics#STRICT}, which retains the
 * association written by the caller. Use {@code RELAXED} or {@code FAST}
 * explicitly when matrix-chain reassociation is acceptable.</p>
 */
public final class MatrixCompiler {
    private MatrixCompiler() {
    }

    public static ExecutionPlan compile(MatrixExpr expression) {
        return compile(expression, OptimizationSemantics.STRICT);
    }

    public static ExecutionPlan compile(MatrixExpr expression,
                                        OptimizationSemantics semantics) {
        return compile(expression, semantics, new FlopCostModel());
    }

    public static ExecutionPlan compile(MatrixExpr expression,
                                        OptimizationSemantics semantics,
                                        MatrixCostModel costModel) {
        return new MatrixOptimizer(semantics, costModel).optimize(expression);
    }

    /**
     * Alias emphasizing that this method returns an inspectable plan.
     */
    public static ExecutionPlan plan(MatrixExpr expression) {
        return compile(expression);
    }

    public static ExecutionPlan plan(MatrixExpr expression,
                                     OptimizationSemantics semantics) {
        return compile(expression, semantics);
    }

    public static ExecutionPlan plan(MatrixExpr expression,
                                     OptimizationSemantics semantics,
                                     MatrixCostModel costModel) {
        return compile(expression, semantics, costModel);
    }

    public static Matrix evaluate(MatrixExpr expression) {
        return compile(expression).evaluate();
    }

    public static Matrix evaluate(MatrixExpr expression,
                                  OptimizationSemantics semantics) {
        return compile(expression, semantics).evaluate();
    }

    public static Matrix evaluate(MatrixExpr expression,
                                  OptimizationSemantics semantics,
                                  MatrixCostModel costModel) {
        return compile(expression, semantics, costModel).evaluate();
    }

    public static Matrix evaluate(ExecutionPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("Execution plan must not be null");
        }
        return plan.evaluate();
    }
}
