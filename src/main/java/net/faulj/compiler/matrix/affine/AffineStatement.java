package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Immutable semantic statement in the M2 affine program.
 */
public final class AffineStatement {
    private final int id;
    private final StatementKind kind;
    private final IterationDomain domain;
    private final List<AffineAccess> accesses;
    private final String computation;
    private final ReductionMetadata reduction;

    public AffineStatement(int id,
                           StatementKind kind,
                           IterationDomain domain,
                           List<AffineAccess> accesses,
                           String computation) {
        this(id, kind, domain, accesses, computation, null);
    }

    public AffineStatement(int id,
                           StatementKind kind,
                           IterationDomain domain,
                           List<AffineAccess> accesses,
                           String computation,
                           ReductionMetadata reduction) {
        if (id < 0) {
            throw new IllegalArgumentException("Statement ID must be non-negative");
        }
        this.id = id;
        this.kind = Objects.requireNonNull(kind, "Statement kind must not be null");
        this.domain = Objects.requireNonNull(domain, "Statement domain must not be null");
        if (accesses == null) {
            throw new IllegalArgumentException("Statement accesses must not be null");
        }
        List<AffineAccess> copy = new ArrayList<>(accesses);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Statement accesses must not contain nulls");
        }
        this.accesses = Collections.unmodifiableList(copy);
        if (computation == null || computation.isBlank()) {
            throw new IllegalArgumentException("Statement computation must not be blank");
        }
        this.computation = computation;
        this.reduction = reduction;
    }

    public int id() {
        return id;
    }

    public String name() {
        return "S" + id;
    }

    public StatementKind kind() {
        return kind;
    }

    public IterationDomain domain() {
        return domain;
    }

    public List<AffineAccess> accesses() {
        return accesses;
    }

    public String computation() {
        return computation;
    }

    public ReductionMetadata reduction() {
        return reduction;
    }

    public ReductionMetadata reductionMetadata() {
        return reduction;
    }

    public boolean hasReduction() {
        return reduction != null;
    }

    public List<AffineAccess> reads() {
        return accesses.stream().filter(AffineAccess::reads).collect(Collectors.toUnmodifiableList());
    }

    public List<AffineAccess> writes() {
        return accesses.stream().filter(AffineAccess::writes).collect(Collectors.toUnmodifiableList());
    }

    @Override
    public String toString() {
        return name() + "[" + domain.variables().stream()
            .map(AffineVariable::name)
            .collect(Collectors.joining(",")) + "] : " + computation;
    }
}
