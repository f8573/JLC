package net.faulj.compiler.matrix;

/**
 * Element-wise addition node in an execution plan.
 */
public final class PlanAdd implements PlanNode {
    private final PlanNode lhs;
    private final PlanNode rhs;

    public PlanAdd(PlanNode lhs, PlanNode rhs) {
        PlanNode.requireNode(lhs, "Plan Add left operand");
        PlanNode.requireNode(rhs, "Plan Add right operand");
        if (!lhs.shape().equals(rhs.shape())) {
            throw new IllegalArgumentException(
                "Cannot plan addition for shapes " + lhs.shape() + " and " + rhs.shape()
                    + ": shapes must be identical");
        }
        this.lhs = lhs;
        this.rhs = rhs;
    }

    public PlanNode lhs() {
        return lhs;
    }

    public PlanNode rhs() {
        return rhs;
    }

    public PlanNode left() {
        return lhs;
    }

    public PlanNode right() {
        return rhs;
    }

    @Override
    public MatrixShape shape() {
        return lhs.shape();
    }
}
