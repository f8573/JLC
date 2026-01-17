package net.faulj.givens;

import net.faulj.matrix.Matrix;

/**
 * Represents a Givens rotation [ c  s ]
 * [ -s c ]
 * Used to zero out elements or apply similarity transformations.
 */
public class GivensRotation {
    public final double c;
    public final double s;
    public final double r; // The norm of the vector being rotated (optional use)

    public GivensRotation(double c, double s, double r) {
        this.c = c;
        this.s = s;
        this.r = r;
    }

    /**
     * Computes a Givens rotation that annihilates the second component of vector [a, b]^T.
     * The result maps [a, b]^T to [r, 0]^T.
     */
    public static GivensRotation compute(double a, double b) {
        double c, s, r;
        if (b == 0) {
            c = 1.0;
            s = 0.0;
            r = a;
        } else if (a == 0) {
            c = 0.0;
            s = 1.0;
            r = b;
        } else {
            double absA = Math.abs(a);
            double absB = Math.abs(b);
            if (absA > absB) {
                double t = b / a;
                double u = Math.signum(a) * Math.sqrt(1 + t * t);
                c = 1 / u;
                s = -c * t; // Note: standard definition often uses s = -b/r, here we ensure consistent sign
                r = a * u;
            } else {
                double t = a / b;
                double u = Math.signum(b) * Math.sqrt(1 + t * t);
                s = -1 / u;
                c = -s * t;
                r = b * u;
            }

            // Re-normalize for numerical safety if needed, though the above is stable.
            // Simplified consistent computation:
            double h = Math.hypot(a, b);
            r = h;
            c = a / h;
            s = -b / h;
        }
        return new GivensRotation(c, s, r);
    }

    /**
     * Applies the rotation to rows i and k of matrix A (from the left).
     * G^T * A
     */
    public void applyLeft(Matrix A, int i, int k, int colStart, int colEnd) {
        for (int col = colStart; col <= colEnd; col++) {
            double valI = A.get(i, col);
            double valK = A.get(k, col);
            A.set(i, col, c * valI - s * valK);
            A.set(k, col, s * valI + c * valK);
        }
    }

    /**
     * Applies the rotation to columns i and k of matrix A (from the right).
     * A * G
     */
    public void applyRight(Matrix A, int i, int k, int rowStart, int rowEnd) {
        for (int row = rowStart; row <= rowEnd; row++) {
            double valI = A.get(row, i);
            double valK = A.get(row, k);
            A.set(row, i, c * valI - s * valK);
            A.set(row, k, s * valI + c * valK);
        }
    }

    public void applyColumn(Matrix A, int colIndex, int i, int k) {
        double valI = A.get(i, colIndex);
        double valK = A.get(k, colIndex);
        A.set(i, colIndex, c * valI - s * valK);
        A.set(k, colIndex, s * valI + c * valK);
    }
}