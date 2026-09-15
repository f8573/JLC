package net.faulj.compiler.matrix;

/**
 * Immutable operation node in an {@link ExecutionPlan}.
 *
 * <p>Plan nodes are intentionally distinct from {@link MatrixExpr} nodes.
 * The optimizer can therefore leave the source graph untouched and produce
 * a separately inspectable selected operation graph.</p>
 */
public sealed interface PlanNode
    permits PlanInput, PlanMatMul, PlanAdd, PlanScale, PlanTranspose {

    MatrixShape shape();

    default MatrixShape outputShape() {
        return shape();
    }

    default int rows() {
        return shape().rows();
    }

    default int columns() {
        return shape().columns();
    }

    static void requireNode(PlanNode node, String role) {
        if (node == null) {
            throw new IllegalArgumentException(role + " must not be null");
        }
    }
}
