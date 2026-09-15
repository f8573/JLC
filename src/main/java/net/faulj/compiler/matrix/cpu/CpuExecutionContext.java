package net.faulj.compiler.matrix.cpu;

import java.util.IdentityHashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;

/** Package-private runtime state for one CPU plan invocation. */
final class CpuExecutionContext {
    private final IdentityHashMap<LogicalBuffer, Matrix> values = new IdentityHashMap<>();
    private final IdentityHashMap<LogicalBuffer, Boolean> owned = new IdentityHashMap<>();

    void bind(LogicalBuffer buffer, Matrix matrix) {
        values.put(buffer, matrix);
    }

    void bindOwned(LogicalBuffer buffer, Matrix matrix) {
        values.put(buffer, matrix);
        owned.put(buffer, Boolean.TRUE);
    }

    Matrix value(LogicalBuffer buffer) {
        Matrix value = values.get(buffer);
        if (value == null) {
            throw new IllegalStateException(
                "CPU step requires an unmaterialized logical buffer %" + buffer.id());
        }
        return value;
    }

    Matrix allocate(LogicalBuffer buffer) {
        Matrix existing = values.get(buffer);
        if (existing != null) {
            return existing;
        }
        Matrix created = new Matrix(buffer.shape().rows(), buffer.shape().columns());
        values.put(buffer, created);
        owned.put(buffer, Boolean.TRUE);
        return created;
    }

    Map<LogicalBuffer, Matrix> values() {
        return values;
    }

    boolean isOwned(LogicalBuffer buffer) {
        return owned.containsKey(buffer);
    }
}
