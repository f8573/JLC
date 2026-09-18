package net.faulj.eigen;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.qr.BlockedHessenbergQR;
import net.faulj.eigen.qr.ImplicitQRFrancis;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.decomposition.result.SVDResult;
import net.faulj.svd.DivideAndConquerSVD;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * Diagnostic test: traces reconstruction errors through each decomposition
 * stage for matrices from 2x2 to 10x10.
 */
public class SmallMatrixDiagnostic {

    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        Vector[] colsV = new Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) col[r] = a[r][c];
            colsV[c] = new Vector(col);
        }
        return new Matrix(colsV);
    }

    private static Matrix randomMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = rnd.nextDouble() * 2 - 1;
        return fromRowMajor(a);
    }

    private static double frobNorm(Matrix M) {
        return M.frobeniusNorm();
    }

    private static double relError(Matrix A, Matrix B) {
        double normA = frobNorm(A);
        if (normA < 1e-14) return frobNorm(A.subtract(B));
        return frobNorm(A.subtract(B)) / normA;
    }

    private static double orthError(Matrix Q) {
        int n = Q.getRowCount();
        return frobNorm(Q.transpose().multiply(Q).subtract(Matrix.Identity(n)));
    }

    @Test
    public void diagnoseSpecificFailing() {
        // 5x5#1 case with seed=5137
        System.out.println("=== Diagnosing 5x5#1 (seed=5137) ===\n");
        Matrix A = randomMatrix(5, 5137);

        SchurResult schur = RealSchurDecomposition.decompose(A);
        Matrix T = schur.getT();
        Matrix U = schur.getU();
        net.faulj.scalar.Complex[] eigs = schur.getEigenvalues();
        Matrix evecs = schur.getEigenvectors();

        System.out.println("T diagonal and subdiagonal:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("  T[%d,%d] = %.16f", i, i, T.get(i, i));
            if (i > 0) System.out.printf("  T[%d,%d] = %.6e", i, i-1, T.get(i, i-1));
            System.out.println();
        }

        System.out.println("\nEigenvalues:");
        for (int i = 0; i < 5; i++)
            System.out.printf("  lambda[%d] = %.10f + %.10fi\n", i, eigs[i].real, eigs[i].imag);

        // Compute residuals in A-space
        System.out.println("\nEigenvector residuals:");
        for (int j = 0; j < 5; j++) {
            double lr = eigs[j].real, li = eigs[j].imag;
            double[] vr = new double[5], vi = new double[5];
            for (int r = 0; r < 5; r++) {
                vr[r] = evecs.get(r, j);
                vi[r] = evecs.getImag(r, j);
            }
            double vNorm = 0;
            for (int r = 0; r < 5; r++) vNorm += vr[r]*vr[r] + vi[r]*vi[r];
            vNorm = Math.sqrt(vNorm);

            double resSq = 0;
            for (int r = 0; r < 5; r++) {
                double avr = 0, avi = 0;
                for (int c = 0; c < 5; c++) {
                    avr += A.get(r, c) * vr[c];
                    avi += A.get(r, c) * vi[c];
                }
                double lvr = lr * vr[r] - li * vi[r];
                double lvi = lr * vi[r] + li * vr[r];
                resSq += (avr - lvr)*(avr - lvr) + (avi - lvi)*(avi - lvi);
            }
            double res = Math.sqrt(resSq) / Math.max(1e-14, vNorm);
            System.out.printf("  j=%d lambda=%.6f+%.6fi ||v||=%.6f res=%.2e\n",
                    j, lr, li, vNorm, res);
        }

        // Compute back-sub eigenvectors directly and check T-space residual
        System.out.println("\nDirect T-space back-sub check:");
        double[] tData = new double[25];
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++)
                tData[r * 5 + c] = T.get(r, c);

        for (int j = 0; j < 5; j++) {
            if (Math.abs(eigs[j].imag) > 1e-10) {
                System.out.printf("  j=%d (complex, skip)\n", j);
                continue;
            }
            double lambda = eigs[j].real;
            double[] y = new double[5];
            // Simple back-sub
            for (int k = 4; k >= 0; k--) {
                if (k == j) {
                    y[k] = 1.0;
                } else {
                    double sum = 0;
                    for (int m = k + 1; m < 5; m++) {
                        sum += tData[k * 5 + m] * y[m];
                    }
                    double diag = tData[k * 5 + k] - lambda;
                    y[k] = (Math.abs(diag) < 1e-16) ? 0.0 : -sum / diag;
                }
            }
            // T-space residual
            double resSq = 0;
            for (int r = 0; r < 5; r++) {
                double sum = 0;
                for (int c = 0; c < 5; c++) sum += tData[r * 5 + c] * y[c];
                double diff = sum - lambda * y[r];
                resSq += diff * diff;
            }
            double yNorm = 0;
            for (double v : y) yNorm += v * v;
            yNorm = Math.sqrt(yNorm);
            System.out.printf("  j=%d lambda=%.6f ||y||=%.6f T-res=%.2e y=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    j, lambda, yNorm, Math.sqrt(resSq) / Math.max(1e-14, yNorm),
                    y[0], y[1], y[2], y[3], y[4]);
        }
    }

    @Test
    public void diagnoseSpecificMatrix() {
        System.out.println("=== Diagnosing [[1,2,3],[4,5,6],[7,8,9]] ===\n");
        Matrix A = fromRowMajor(new double[][]{{1,2,3},{4,5,6},{7,8,9}});

        // Step 1: Hessenberg
        HessenbergResult hess = BlockedHessenbergQR.decompose(A);
        Matrix H = hess.getH();
        Matrix Q = hess.getQ();
        double hessRecon = relError(A, Q.multiply(H).multiply(Q.transpose()));
        double hessOrth = orthError(Q);
        System.out.printf("Hessenberg: orthError=%.2e  reconError=%.2e\n", hessOrth, hessRecon);

        // Check Hessenberg structure
        boolean isHess = true;
        for (int i = 2; i < 3; i++)
            for (int j = 0; j < i - 1; j++)
                if (Math.abs(H.get(i, j)) > 1e-14) isHess = false;
        System.out.println("Is Hessenberg: " + isHess);
        System.out.println("H = ");
        printMatrix(H);
        System.out.println("Q = ");
        printMatrix(Q);

        // Step 2: Schur
        SchurResult schur = RealSchurDecomposition.decompose(A);
        Matrix T = schur.getT();
        Matrix U = schur.getU();
        double schurRecon = relError(A, U.multiply(T).multiply(U.transpose()));
        double schurOrth = orthError(U);
        System.out.printf("\nSchur: orthError=%.2e  reconError=%.2e\n", schurOrth, schurRecon);
        System.out.println("T = ");
        printMatrix(T);

        // Step 3: eigenvalues
        net.faulj.scalar.Complex[] eigs = schur.getEigenvalues();
        System.out.println("\nEigenvalues from getEigenvalues():");
        for (int i = 0; i < eigs.length; i++) {
            System.out.printf("  lambda[%d] = %.10f + %.10fi\n", i, eigs[i].real, eigs[i].imag);
        }
        System.out.println("T diagonal:");
        for (int i = 0; i < 3; i++) {
            System.out.printf("  T[%d,%d] = %.16f\n", i, i, T.get(i, i));
        }
        System.out.println("T subdiagonal:");
        for (int i = 1; i < 3; i++) {
            System.out.printf("  T[%d,%d] = %.16e\n", i, i-1, T.get(i, i-1));
        }

        // Expected eigenvalues of [[1,2,3],[4,5,6],[7,8,9]]:
        // 16.11684..., -1.11684..., 0
        double expectedTrace = 15.0;
        double actualTrace = T.trace();
        System.out.printf("\nTrace: expected=%.6f  actual=%.6f  error=%.2e\n",
                expectedTrace, actualTrace, Math.abs(expectedTrace - actualTrace));

        // Step 4: Eigenvectors
        Matrix evecs = schur.getEigenvectors();
        if (evecs != null) {
            System.out.println("\nEigenvectors:");
            printMatrix(evecs);

            // Check A*v = lambda*v for each eigenvector
            for (int j = 0; j < 3; j++) {
                double lambda = eigs[j].real;
                double[] v = new double[3];
                for (int r = 0; r < 3; r++) v[r] = evecs.get(r, j);
                double vNorm = 0;
                for (double x : v) vNorm += x * x;
                vNorm = Math.sqrt(vNorm);

                double[] Av = new double[3];
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        Av[r] += A.get(r, c) * v[c];

                double resNorm = 0;
                for (int r = 0; r < 3; r++) {
                    double diff = Av[r] - lambda * v[r];
                    resNorm += diff * diff;
                }
                resNorm = Math.sqrt(resNorm);
                System.out.printf("  lambda=%.6f ||v||=%.6f ||Av-lv||=%.2e\n",
                        lambda, vNorm, resNorm);
            }
        }

        assertTrue("Schur reconstruction error too large: " + schurRecon, schurRecon < 1e-10);
    }

    @Test
    public void diagnoseAllSmallSizes() {
        System.out.println("=== Decomposition Diagnostics: 2x2 to 10x10 ===\n");
        System.out.printf("%-5s | %-12s %-12s | %-12s %-12s | %-12s %-12s | %-12s\n",
                "Size", "Hess.Orth", "Hess.Recon", "Schur.Orth", "Schur.Recon",
                "SVD.U.Orth", "SVD.Recon", "Eig.MaxRes");
        System.out.println("-".repeat(110));

        // Test 25+ matrices: 3 per size from 2 to 10
        for (int n = 2; n <= 10; n++) {
            for (int trial = 0; trial < 3; trial++) {
                long seed = 1000 * n + trial * 137;
                Matrix A = randomMatrix(n, seed);

                // Hessenberg
                double hessOrth = 0, hessRecon = 0;
                try {
                    HessenbergResult hess = BlockedHessenbergQR.decompose(A);
                    hessOrth = orthError(hess.getQ());
                    hessRecon = relError(A, hess.getQ().multiply(hess.getH()).multiply(hess.getQ().transpose()));
                } catch (Exception e) {
                    hessOrth = hessRecon = Double.NaN;
                }

                // Schur
                double schurOrth = 0, schurRecon = 0;
                try {
                    SchurResult schur = RealSchurDecomposition.decompose(A);
                    schurOrth = orthError(schur.getU());
                    schurRecon = relError(A, schur.getU().multiply(schur.getT()).multiply(schur.getU().transpose()));
                } catch (Exception e) {
                    schurOrth = schurRecon = Double.NaN;
                }

                // SVD
                double svdUOrth = 0, svdRecon = 0;
                try {
                    DivideAndConquerSVD svdSolver = new DivideAndConquerSVD();
                    SVDResult svd = svdSolver.decompose(A);
                    Matrix Ufull = svd.getU();
                    Matrix S = svd.getSigma();
                    Matrix Vt = svd.getV().transpose();
                    svdUOrth = orthError(Ufull);
                    svdRecon = relError(A, Ufull.multiply(S).multiply(Vt));
                } catch (Exception e) {
                    svdUOrth = svdRecon = Double.NaN;
                }

                // Eigenvector residual
                double eigMaxRes = 0;
                try {
                    SchurResult schur = RealSchurDecomposition.decompose(A);
                    Matrix evecs = schur.getEigenvectors();
                    net.faulj.scalar.Complex[] evals = schur.getEigenvalues();
                    if (evecs != null) {
                        for (int j = 0; j < n; j++) {
                            double lr = evals[j].real;
                            double li = evals[j].imag;
                            double[] vr = new double[n];
                            double[] vi = new double[n];
                            for (int r = 0; r < n; r++) {
                                vr[r] = evecs.get(r, j);
                                vi[r] = evecs.getImag(r, j);
                            }
                            double vNorm = 0;
                            for (int r = 0; r < n; r++) vNorm += vr[r]*vr[r] + vi[r]*vi[r];
                            vNorm = Math.sqrt(vNorm);
                            if (vNorm < 1e-14) { eigMaxRes = Double.MAX_VALUE; continue; }

                            // Compute A*(vr+i*vi) - (lr+i*li)*(vr+i*vi)
                            double resSq = 0;
                            for (int r = 0; r < n; r++) {
                                double avr = 0, avi = 0;
                                for (int c = 0; c < n; c++) {
                                    avr += A.get(r, c) * vr[c];
                                    avi += A.get(r, c) * vi[c];
                                }
                                double lvr = lr * vr[r] - li * vi[r];
                                double lvi = lr * vi[r] + li * vr[r];
                                double dr = avr - lvr;
                                double di = avi - lvi;
                                resSq += dr*dr + di*di;
                            }
                            double res = Math.sqrt(resSq) / vNorm;
                            eigMaxRes = Math.max(eigMaxRes, res);
                        }
                    }
                } catch (Exception e) {
                    eigMaxRes = Double.NaN;
                }

                System.out.printf("%dx%d#%d | %.4e  %.4e  | %.4e  %.4e  | %.4e  %.4e  | %.4e\n",
                        n, n, trial, hessOrth, hessRecon, schurOrth, schurRecon,
                        svdUOrth, svdRecon, eigMaxRes);
            }
        }
    }

    private static void printMatrix(Matrix m) {
        for (int r = 0; r < m.getRowCount(); r++) {
            System.out.print("  [");
            for (int c = 0; c < m.getColumnCount(); c++) {
                System.out.printf("%12.6f", m.get(r, c));
                if (c < m.getColumnCount() - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
}
