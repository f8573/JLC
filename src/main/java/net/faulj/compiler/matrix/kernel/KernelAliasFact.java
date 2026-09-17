package net.faulj.compiler.matrix.kernel;

import java.util.Objects;

import net.faulj.compiler.matrix.affine.AliasRelation;

/** One conservative alias fact between two kernel buffers. */
public final class KernelAliasFact {
    private final KernelBuffer first;
    private final KernelBuffer second;
    private final AliasRelation relation;

    public KernelAliasFact(KernelBuffer first,
                           KernelBuffer second,
                           AliasRelation relation) {
        this.first = Objects.requireNonNull(first, "First alias buffer must not be null");
        this.second = Objects.requireNonNull(second, "Second alias buffer must not be null");
        this.relation = Objects.requireNonNull(relation, "Alias relation must not be null");
    }

    public KernelBuffer first() {
        return first;
    }

    public KernelBuffer second() {
        return second;
    }

    public AliasRelation relation() {
        return relation;
    }

    @Override
    public String toString() {
        return "%" + first.id() + " <-> %" + second.id() + " = " + relation;
    }
}
