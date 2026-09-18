package net.faulj.compiler.matrix.cpu;

/** Runtime value-lane requirement used by physical storage compatibility. */
public enum StorageValueKind {
    REAL,
    COMPLEX,
    UNKNOWN;

    public boolean isKnown() {
        return this != UNKNOWN;
    }

    static StorageValueKind fromMatrix(net.faulj.matrix.Matrix matrix) {
        if (matrix == null) {
            return UNKNOWN;
        }
        return matrix.hasImagData() ? COMPLEX : REAL;
    }

    static StorageValueKind merge(StorageValueKind first, StorageValueKind second) {
        if (first == COMPLEX || second == COMPLEX) {
            return COMPLEX;
        }
        if (first == UNKNOWN || second == UNKNOWN) {
            return UNKNOWN;
        }
        return REAL;
    }
}
