package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import net.faulj.compiler.matrix.affine.AffineExpr;

/** Explicit affine address of one kernel buffer. */
public final class KernelAccess {
    private final KernelBuffer buffer;
    private final List<AffineExpr> indices;

    public KernelAccess(KernelBuffer buffer, List<AffineExpr> indices) {
        this.buffer = Objects.requireNonNull(buffer, "Kernel access buffer must not be null");
        if (indices == null || indices.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Kernel access indices must not be null");
        }
        this.indices = Collections.unmodifiableList(new ArrayList<>(indices));
    }

    public static KernelAccess of(KernelBuffer buffer, List<AffineExpr> indices) {
        return new KernelAccess(buffer, indices);
    }

    public KernelBuffer buffer() {
        return buffer;
    }

    public List<AffineExpr> indices() {
        return indices;
    }

    public AffineExpr index(int dimension) {
        return indices.get(dimension);
    }

    public int rank() {
        return indices.size();
    }

    @Override
    public String toString() {
        return "%" + buffer.id() + "["
            + indices.stream().map(AffineExpr::toString).collect(Collectors.joining(","))
            + "]";
    }
}
