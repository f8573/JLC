package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Flat, ordered SSA block used as the body of a v1 kernel. */
public final class KernelBlock {
    private final List<KernelOp> operations;

    public KernelBlock(List<KernelOp> operations) {
        if (operations == null || operations.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Kernel operations must not be null");
        }
        this.operations = Collections.unmodifiableList(new ArrayList<>(operations));
    }

    public List<KernelOp> operations() {
        return operations;
    }

    public List<KernelOp> ops() {
        return operations;
    }
}
