package net.faulj.compiler.matrix.affine;

import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;

/**
 * Public M2 entry point for lowering an M1 plan into semantic affine IR.
 *
 * <p>The overload accepting an expression still goes through
 * {@link MatrixCompiler}; callers cannot accidentally bypass the M1
 * {@link ExecutionPlan} boundary.</p>
 */
public final class MatrixAffineCompiler {
    private MatrixAffineCompiler() {
    }

    public static AffineProgram lower(ExecutionPlan plan) {
        return AffineLowerer.lower(plan);
    }

    public static AffineProgram lower(MatrixExpr expression) {
        return AffineLowerer.lower(expression);
    }

    public static AffineProgram analyze(ExecutionPlan plan) {
        return lower(plan);
    }
}
