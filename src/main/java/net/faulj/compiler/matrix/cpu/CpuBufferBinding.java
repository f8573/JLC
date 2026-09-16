package net.faulj.compiler.matrix.cpu;

import java.util.Objects;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;

/**
 * Binding for one logical external buffer.
 *
 * <p>The matrix reference is borrowed, never copied. A null matrix is useful
 * for an inspectable, plan-only lowering and is rejected if execution is
 * attempted without a separate runtime binding.</p>
 */
public final class CpuBufferBinding {
    private final LogicalBuffer buffer;
    private final Matrix matrix;

    public CpuBufferBinding(LogicalBuffer buffer, Matrix matrix) {
        this.buffer = Objects.requireNonNull(buffer, "CPU buffer must not be null");
        if (!buffer.isExternalInput()) {
            throw new IllegalArgumentException(
                "CPU runtime bindings must target EXTERNAL_INPUT buffers: %" + buffer.id());
        }
        if (matrix != null
            && (matrix.getRowCount() != buffer.shape().rows()
                || matrix.getColumnCount() != buffer.shape().columns())) {
            throw new IllegalArgumentException(
                "Matrix shape " + matrix.getRowCount() + "x" + matrix.getColumnCount()
                    + " does not match logical buffer %" + buffer.id()
                    + " shape " + buffer.shape());
        }
        this.matrix = matrix;
    }

    public LogicalBuffer buffer() {
        return buffer;
    }

    public LogicalBuffer logicalBuffer() {
        return buffer;
    }

    /** The borrowed runtime matrix, or null for an unresolved inspection binding. */
    public Matrix matrix() {
        return matrix;
    }

    public boolean isResolved() {
        return matrix != null;
    }

    @Override
    public String toString() {
        return "%" + buffer.id() + " -> " + buffer.name()
            + (matrix == null ? " (unbound)" : " (borrowed runtime matrix)");
    }
}
