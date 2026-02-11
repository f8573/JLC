package net.faulj.autotune.persist;

import net.faulj.compute.BlockedMultiply;
import net.faulj.matrix.Matrix;

import java.util.Optional;

/**
 * Validate loaded profiles for fingerprint and lightweight sanity.
 */
public final class ProfileValidator {

    private ProfileValidator() {}

    public static boolean validate(TuningProfile prof, MachineFingerprint current, boolean runSanity) {
        if (prof == null || current == null) return false;
        if (!current.getHash().equals(prof.fingerprint)) {
            return false;
        }
        if (prof.schemaVersion == null) return false;

        if (runSanity) {
            try {
                // lightweight deterministic correctness check (small matrices)
                int n = 8;
                double[][] a = new double[n][n];
                double[][] b = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        a[i][j] = ((i + 1) * 31 + (j + 7)) % 17;
                        b[i][j] = ((i + 3) * 13 + (j + 11)) % 19;
                    }
                }
                Matrix A = new Matrix(a);
                Matrix B = new Matrix(b);
                Matrix expected = BlockedMultiply.multiplyNaive(A, B);
                Matrix actual = BlockedMultiply.multiply(A, B);
                double tol = 1e-9;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double e = expected.get(i, j);
                        double r = actual.get(i, j);
                        if (Math.abs(e - r) > tol) return false;
                    }
                }
            } catch (Throwable t) {
                return false;
            }
        }
        return true;
    }
}
