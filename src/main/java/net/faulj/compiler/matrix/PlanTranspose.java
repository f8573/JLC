package net.faulj.compiler.matrix;

/**
 * Transpose node in an execution plan.
 */
public final class PlanTranspose implements PlanNode {
    private final PlanNode operand;
    private final MatrixShape shape;

    public PlanTranspose(PlanNode operand) {
        PlanNode.requireNode(operand, "Plan Transpose operand");
        this.operand = operand;
        this.shape = new MatrixShape(operand.shape().columns(), operand.shape().rows());
    }

    public PlanNode operand() {
        return operand;
    }

    @Override
    public MatrixShape shape() {
        return shape;
    }
}
