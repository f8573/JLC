package net.faulj.compiler.matrix.kernel;

import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;

/** One explicit rectangular loop in a portable kernel loop nest. */
public final class KernelLoop {
    private final AffineVariable inductionVariable;
    private final AffineVariable semanticVariable;
    private final long lowerBound;
    private final long upperBound;
    private final long step;
    private final AffineExpr valueExpression;

    /** Create a canonical loop whose induction variable is semantic. */
    public KernelLoop(AffineVariable variable, long lowerBound, long upperBound) {
        this(variable, variable, lowerBound, upperBound, 1L,
            AffineExpr.variable(variable));
    }

    public KernelLoop(AffineVariable inductionVariable,
                      AffineVariable semanticVariable,
                      long lowerBound,
                      long upperBound,
                      long step,
                      AffineExpr valueExpression) {
        this.inductionVariable = Objects.requireNonNull(
            inductionVariable, "Kernel induction variable must not be null");
        this.semanticVariable = Objects.requireNonNull(
            semanticVariable, "Kernel semantic variable must not be null");
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.step = step;
        this.valueExpression = valueExpression;
    }

    public AffineVariable inductionVariable() {
        return inductionVariable;
    }

    public AffineVariable variable() {
        return inductionVariable;
    }

    public AffineVariable semanticVariable() {
        return semanticVariable;
    }

    public long lowerBound() {
        return lowerBound;
    }

    public long upperBound() {
        return upperBound;
    }

    public long step() {
        return step;
    }

    /** Binding for the semantic variable; null is not executable in R3 v1. */
    public AffineExpr valueExpression() {
        return valueExpression;
    }

    public boolean isCanonical() {
        return semanticVariable.equals(inductionVariable)
            && step == 1L
            && valueExpression != null
            && valueExpression.equals(AffineExpr.variable(inductionVariable));
    }

    @Override
    public String toString() {
        return "for " + inductionVariable + " = " + lowerBound + ".."
            + upperBound + " step " + step;
    }
}
