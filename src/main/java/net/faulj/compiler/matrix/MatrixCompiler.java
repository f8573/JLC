package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;
import net.faulj.compiler.matrix.cpu.FusionStrategy;

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

    /**
     * Compile an expression through the complete inspectable M1-M4 pipeline.
     * The returned program executes its lowered CPU plan when requested.
     */
    public static CompiledMatrixProgram compileProgram(MatrixExpr expression) {
        return compileProgram(expression, OptimizationSemantics.STRICT);
    }

    public static CompiledMatrixProgram compileProgram(MatrixExpr expression,
                                                       OptimizationSemantics semantics) {
        return compileProgram(expression, semantics, new FlopCostModel());
    }

    public static CompiledMatrixProgram compileProgram(MatrixExpr expression,
                                                       OptimizationSemantics semantics,
                                                       FusionStrategy fusionStrategy) {
        return compileProgram(expression, semantics, new FlopCostModel(), fusionStrategy);
    }

    public static CompiledMatrixProgram compileProgram(MatrixExpr expression,
                                                       FusionStrategy fusionStrategy) {
        return compileProgram(expression, OptimizationSemantics.STRICT, fusionStrategy);
    }

    public static CompiledMatrixProgram compileProgram(MatrixExpr expression,
                                                       OptimizationSemantics semantics,
                                                       MatrixCostModel costModel) {
        return CompiledMatrixProgram.from(compile(expression, semantics, costModel));
    }

    /** Compile a program with an explicit CPU fusion implementation. */
    public static CompiledMatrixProgram compileProgram(MatrixExpr expression,
                                                       OptimizationSemantics semantics,
                                                       MatrixCostModel costModel,
                                                       FusionStrategy fusionStrategy) {
        if (fusionStrategy == null) {
            throw new IllegalArgumentException("Fusion strategy must not be null");
        }
        return CompiledMatrixProgram.fromWithFusion(
            compile(expression, semantics, costModel), fusionStrategy);
    }
}
