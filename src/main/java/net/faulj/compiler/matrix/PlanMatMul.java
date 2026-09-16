package net.faulj.compiler.matrix;

/**
 * Matrix multiplication node in an execution plan.
 */
public final class PlanMatMul implements PlanNode {
    private final PlanNode lhs;
    private final PlanNode rhs;
    private final MatrixShape shape;

    public PlanMatMul(PlanNode lhs, PlanNode rhs) {
        PlanNode.requireNode(lhs, "Plan MatMul left operand");
        PlanNode.requireNode(rhs, "Plan MatMul right operand");
        if (lhs.shape().columns() != rhs.shape().rows()) {
            throw new IllegalArgumentException(
                "Cannot plan multiplication for shapes " + lhs.shape() + " and " + rhs.shape()
                    + ": left columns (" + lhs.shape().columns()
                    + ") must equal right rows (" + rhs.shape().rows() + ")");
        }
        this.lhs = lhs;
        this.rhs = rhs;
        this.shape = new MatrixShape(lhs.shape().rows(), rhs.shape().columns());
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
        return shape;
    }
}
