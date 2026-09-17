package net.faulj.compiler.matrix.cpu;

import java.util.Objects;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.MemorySpace;

/**
 * Compatibility key for one physical matrix representation.
 *
 * <p>R1 intentionally keeps the key exact: memory space and real/complex
 * lane layout must agree, and shape is checked separately by the physical
 * slot. Unknown facts never become a license to alias known storage.</p>
 */
public record PhysicalStorageClass(MemorySpace memorySpace,
                                   StorageValueKind valueKind) {
    public PhysicalStorageClass {
        Objects.requireNonNull(memorySpace, "Physical memory space must not be null");
        Objects.requireNonNull(valueKind, "Physical value kind must not be null");
    }

    public static PhysicalStorageClass fromMatrix(net.faulj.matrix.Matrix matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("Matrix must not be null");
        }
        return new PhysicalStorageClass(
            MemorySpace.from(matrix), StorageValueKind.fromMatrix(matrix));
    }

    public boolean isKnown() {
        return memorySpace != MemorySpace.UNKNOWN && valueKind.isKnown();
    }

    public boolean compatibleWith(PhysicalStorageClass other) {
        return other != null
            && isKnown()
            && other.isKnown()
            && memorySpace == other.memorySpace
            && valueKind == other.valueKind;
    }

    /** Stable human-readable compatibility key used by plan dumps. */
    public String key() {
        return memorySpace.name().toLowerCase() + "-"
            + valueKind.name().toLowerCase();
    }

    /**
     * Payload bytes for an exact-shape slot. Unknown values use the real lane
     * as a diagnostic lower bound; executable plans reclassify from runtime
     * bindings before allocating.
     */
    public long payloadBytes(MatrixShape shape) {
        if (shape == null) {
            throw new IllegalArgumentException("Physical shape must not be null");
        }
        long elements;
        try {
            elements = Math.multiplyExact((long) shape.rows(), (long) shape.columns());
            long lanes = valueKind == StorageValueKind.COMPLEX ? 2L : 1L;
            return Math.multiplyExact(Math.multiplyExact(elements, Double.BYTES), lanes);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    @Override
    public String toString() {
        return key();
    }
}
