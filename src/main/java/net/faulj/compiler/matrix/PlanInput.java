package net.faulj.compiler.matrix;

import net.faulj.matrix.Matrix;

/**
 * Input leaf in an execution plan. The source can be a runtime or symbolic
 * input; symbolic plans are inspectable but not executable.
 */
public final class PlanInput implements PlanNode {
    private final MatrixExpr source;

    public PlanInput(MatrixExpr source) {
        if (!(source instanceof Input) && !(source instanceof SymbolicInput)) {
            throw new IllegalArgumentException(
                "PlanInput source must be an Input or SymbolicInput, got "
                    + (source == null ? "null" : source.getClass().getSimpleName()));
        }
        this.source = source;
    }

    public MatrixExpr source() {
        return source;
    }

    public boolean isExecutable() {
        return source instanceof Input;
    }

    public Input runtimeInput() {
        if (!(source instanceof Input input)) {
            throw new IllegalStateException("Symbolic plan input has no runtime Matrix");
        }
        return input;
    }

    public Matrix matrix() {
        return runtimeInput().matrix();
    }

    public String name() {
        if (source instanceof Input input) {
            return input.name();
        }
        return ((SymbolicInput) source).name();
    }

    @Override
    public MatrixShape shape() {
        return source.shape();
    }
}
