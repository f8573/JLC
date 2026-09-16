package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * One affine memory access to a logical matrix buffer.
 */
public final class AffineAccess {
    private final LogicalBuffer buffer;
    private final AccessKind kind;
    private final List<AffineExpr> indices;

    public AffineAccess(LogicalBuffer buffer, AccessKind kind, List<AffineExpr> indices) {
        this.buffer = Objects.requireNonNull(buffer, "Access buffer must not be null");
        this.kind = Objects.requireNonNull(kind, "Access kind must not be null");
        if (indices == null) {
            throw new IllegalArgumentException("Access indices must not be null");
        }
        List<AffineExpr> copy = new ArrayList<>(indices);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Access indices must not contain nulls");
        }
        this.indices = Collections.unmodifiableList(copy);
    }

    public static AffineAccess read(LogicalBuffer buffer, AffineExpr... indices) {
        return new AffineAccess(buffer, AccessKind.READ, List.of(indices));
    }

    public static AffineAccess write(LogicalBuffer buffer, AffineExpr... indices) {
        return new AffineAccess(buffer, AccessKind.WRITE, List.of(indices));
    }

    public static AffineAccess readWrite(LogicalBuffer buffer, AffineExpr... indices) {
        return new AffineAccess(buffer, AccessKind.READ_WRITE, List.of(indices));
    }

    public static AffineAccess reduction(LogicalBuffer buffer, AffineExpr... indices) {
        return new AffineAccess(buffer, AccessKind.REDUCTION, List.of(indices));
    }

    public LogicalBuffer buffer() {
        return buffer;
    }

    public AccessKind kind() {
        return kind;
    }

    public List<AffineExpr> indices() {
        return indices;
    }

    public AffineExpr index(int dimension) {
        return indices.get(dimension);
    }

    public boolean reads() {
        return kind.reads();
    }

    public boolean writes() {
        return kind.writes();
    }

    public boolean isReduction() {
        return kind.isReduction();
    }

    public boolean hasSameIndices(AffineAccess other) {
        return other != null && indices.equals(other.indices);
    }

    public String location() {
        return "%" + buffer.id() + formatIndices();
    }

    private String formatIndices() {
        return "[" + indices.stream().map(AffineExpr::toString).collect(Collectors.joining(",")) + "]";
    }

    @Override
    public String toString() {
        return kind + " " + location();
    }
}
