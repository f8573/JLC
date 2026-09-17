package net.faulj.compiler.matrix;

/**
 * Scalar multiplication node in an execution plan.
 */
public final class PlanScale implements PlanNode {
    private final double factor;
    private final PlanNode operand;

    public PlanScale(double factor, PlanNode operand) {
        PlanNode.requireNode(operand, "Plan Scale operand");
        this.factor = factor;
        this.operand = operand;
    }

    public PlanScale(PlanNode operand, double factor) {
        this(factor, operand);
    }

    public double factor() {
        return factor;
    }

    public double scalar() {
        return factor;
    }

    public PlanNode operand() {
        return operand;
    }

    @Override
    public MatrixShape shape() {
        return operand.shape();
    }
}
