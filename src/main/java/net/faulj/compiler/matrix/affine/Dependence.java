package net.faulj.compiler.matrix.affine;

import java.util.Objects;

/**
 * One ordered dependence fact or conservative uncertainty record.
 */
public final class Dependence {
    private final AffineStatement source;
    private final AffineStatement sink;
    private final DependenceKind kind;
    private final DependenceStatus status;
    private final String relation;
    private final ReductionMetadata reduction;

    public Dependence(AffineStatement source,
                      AffineStatement sink,
                      DependenceKind kind,
                      DependenceStatus status,
                      String relation) {
        this(source, sink, kind, status, relation, null);
    }

    public Dependence(AffineStatement source,
                      AffineStatement sink,
                      DependenceKind kind,
                      DependenceStatus status,
                      String relation,
                      ReductionMetadata reduction) {
        this.source = Objects.requireNonNull(source, "Dependence source must not be null");
        this.sink = Objects.requireNonNull(sink, "Dependence sink must not be null");
        this.kind = Objects.requireNonNull(kind, "Dependence kind must not be null");
        this.status = Objects.requireNonNull(status, "Dependence status must not be null");
        if (relation == null || relation.isBlank()) {
            throw new IllegalArgumentException("Dependence relation must not be blank");
        }
        this.relation = relation;
        this.reduction = reduction;
    }

    public AffineStatement source() {
        return source;
    }

    public AffineStatement sink() {
        return sink;
    }

    public AffineStatement from() {
        return source;
    }

    public AffineStatement to() {
        return sink;
    }

    public DependenceKind kind() {
        return kind;
    }

    public DependenceStatus status() {
        return status;
    }

    public String relation() {
        return relation;
    }

    public ReductionMetadata reduction() {
        return reduction;
    }

    public ReductionMetadata reductionMetadata() {
        return reduction;
    }

    public boolean isProven() {
        return status == DependenceStatus.PROVEN_DEPENDENCE;
    }

    public boolean isUnknown() {
        return status == DependenceStatus.UNKNOWN;
    }

    @Override
    public String toString() {
        String metadata = reduction == null ? "" : " reduction=" + reduction;
        return source.name() + " -> " + sink.name() + " : " + kind + " [" + status + "] "
            + relation + metadata;
    }
}
