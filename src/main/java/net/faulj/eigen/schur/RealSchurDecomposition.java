package net.faulj.eigen.schur;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.matrix.Matrix;
import net.faulj.core.Tolerance;

public class RealSchurDecomposition {

    /**
     * Computes the Real Schur Decomposition of a square matrix A.
     * A = U * T * U^T
     *
     * @param A The matrix to decompose.
     * @return The SchurResult containing T, U, and eigenvalues.
     */
    public static SchurResult decompose(Matrix A) {
        if (!A.isSquare()) {
            throw new IllegalArgumentException("Schur decomposition requires a square matrix.");
        }

        // 1. Hessenberg Reduction: A = Q * H * Q^T
        // The existing HessenbergReduction returns {H, Q} (where Q is accumulated V vectors or explicit)
        // Assuming current implementation returns explicit matrices [H, Q].
        Matrix[] hessenberg = HessenbergReduction.decompose(A);
        Matrix H = hessenberg[0];
        Matrix Q = hessenberg[1];

        // 2. Implicit QR Iteration (Francis Double Shift): H = Z * T * Z^T
        // We pass Q into the QR algorithm so it accumulates the updates: U = Q * Z
        SchurResult qrResult = ImplicitQRFrancis.process(H, Q);

        return qrResult;
    }
}