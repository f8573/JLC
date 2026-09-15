package net.faulj.compiler.matrix.affine;

import java.util.Objects;

import net.faulj.compiler.matrix.MatrixShape;

/**
 * Stable logical buffer identity in an {@link AffineProgram}.
 *
 * <p>A logical buffer is not a runtime allocation. In particular, an owned
 * temporary records compiler ownership semantics without allocating, reusing,
 * or freeing a JLC {@code Matrix}.</p>
 */
public final class LogicalBuffer {
    private final int id;
    private final String name;
    private final BufferKind kind;
    private final BufferOwnership ownership;
    private final MemorySpace memorySpace;
    private final MatrixShape shape;
    private final BufferLifetime lifetime;

    public LogicalBuffer(int id,
                         String name,
                         BufferKind kind,
                         BufferOwnership ownership,
                         MemorySpace memorySpace,
                         MatrixShape shape) {
        this(id, name, kind, ownership, memorySpace, shape, BufferLifetime.none());
    }

    public LogicalBuffer(int id,
                         String name,
                         BufferKind kind,
                         BufferOwnership ownership,
                         MemorySpace memorySpace,
                         MatrixShape shape,
                         BufferLifetime lifetime) {
        if (id < 0) {
            throw new IllegalArgumentException("Buffer ID must be non-negative");
        }
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Buffer name must not be blank");
        }
        this.id = id;
        this.name = name;
        this.kind = Objects.requireNonNull(kind, "Buffer kind must not be null");
        this.ownership = Objects.requireNonNull(ownership, "Buffer ownership must not be null");
        this.memorySpace = Objects.requireNonNull(memorySpace, "Memory space must not be null");
        this.shape = Objects.requireNonNull(shape, "Buffer shape must not be null");
        this.lifetime = Objects.requireNonNull(lifetime, "Buffer lifetime must not be null");
        validateCategory();
    }

    public int id() {
        return id;
    }

    public String name() {
        return name;
    }

    public String displayName() {
        return name;
    }

    public BufferKind kind() {
        return kind;
    }

    public BufferOwnership ownership() {
        return ownership;
    }

    public MemorySpace memorySpace() {
        return memorySpace;
    }

    public MatrixShape shape() {
        return shape;
    }

    public BufferLifetime lifetime() {
        return lifetime;
    }

    public boolean isExternalInput() {
        return kind == BufferKind.EXTERNAL_INPUT;
    }

    public boolean isTemporary() {
        return kind == BufferKind.TEMPORARY;
    }

    public boolean isSymbolic() {
        return kind == BufferKind.SYMBOLIC;
    }

    private void validateCategory() {
        if (kind == BufferKind.EXTERNAL_INPUT && ownership != BufferOwnership.BORROWED) {
            throw new IllegalArgumentException("External inputs must be BORROWED");
        }
        if (kind == BufferKind.TEMPORARY && ownership != BufferOwnership.OWNED) {
            throw new IllegalArgumentException("Temporaries must be OWNED");
        }
        if (kind == BufferKind.SYMBOLIC && ownership != BufferOwnership.NONE) {
            throw new IllegalArgumentException("Symbolic buffers must have NONE ownership");
        }
    }

    @Override
    public String toString() {
        return "%" + id + " " + name + " [" + kind + ", " + ownership + ", "
            + memorySpace + ", shape=" + shape + "]";
    }
}
