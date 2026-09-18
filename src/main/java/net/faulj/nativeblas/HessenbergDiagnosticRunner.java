package net.faulj.nativeblas;

import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;

import java.util.Random;

public final class HessenbergDiagnosticRunner {
    private HessenbergDiagnosticRunner() {
    }

    public static void main(String[] args) {
        int[] sizes = {50, 100, 200};
        for (int n : sizes) {
            Matrix a = randomMatrix(n, 900L + n);
            HessenbergResult result = HessenbergReduction.decompose(a);
            Matrix q = result.getQ();
            Matrix h = result.getH();

            double rawOrth = orthogonalityErrorRaw(q);
            double rawRecon = hessenbergReconstructionErrorRaw(a, q, h);
            double below = belowSubdiagonalMax(h);

            System.out.printf("n=%d rawOrth=%.6e rawRecon=%.6e below=%.6e%n",
                n, rawOrth, rawRecon, below);

            try {
                double gemmOrth = q.transpose().multiply(q).subtract(Matrix.Identity(n)).frobeniusNorm();
                double gemmRecon = a.subtract(q.multiply(h).multiply(q.transpose())).frobeniusNorm() / a.frobeniusNorm();
                System.out.printf("n=%d gemmOrth=%.6e gemmRecon=%.6e%n", n, gemmOrth, gemmRecon);
            } catch (Throwable t) {
                System.out.printf("n=%d gemmPathFailed=%s: %s%n",
                    n, t.getClass().getSimpleName(), t.getMessage());
            }
        }
    }

    private static Matrix randomMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        return fromRowMajor(a);
    }

    private static Matrix fromRowMajor(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        Vector[] colsV = new Vector[cols];
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) {
                col[r] = a[r][c];
            }
            colsV[c] = new Vector(col);
        }
        return new Matrix(colsV);
    }

    private static double orthogonalityErrorRaw(Matrix qMatrix) {
        int n = qMatrix.getRowCount();
        double[] q = qMatrix.getRawData();
        double errorSq = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++) {
                    dot += q[k * n + i] * q[k * n + j];
                }
                if (i == j) {
                    dot -= 1.0;
                }
                errorSq += dot * dot;
            }
        }
        return Math.sqrt(errorSq);
    }

    private static double hessenbergReconstructionErrorRaw(Matrix aMatrix, Matrix qMatrix, Matrix hMatrix) {
        int n = aMatrix.getRowCount();
        double[] a = aMatrix.getRawData();
        double[] q = qMatrix.getRawData();
        double[] h = hMatrix.getRawData();
        double[] qh = new double[n * n];
        double[] recon = new double[n * n];

        for (int i = 0; i < n; i++) {
            int qhBase = i * n;
            for (int k = 0; k < n; k++) {
                double qik = q[qhBase + k];
                int hBase = k * n;
                for (int j = 0; j < n; j++) {
                    qh[qhBase + j] += qik * h[hBase + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            int reconBase = i * n;
            int qhBase = i * n;
            for (int k = 0; k < n; k++) {
                double qhik = qh[qhBase + k];
                for (int j = 0; j < n; j++) {
                    recon[reconBase + j] += qhik * q[j * n + k];
                }
            }
        }

        double normA2 = 0.0;
        double diff2 = 0.0;
        for (int idx = 0; idx < a.length; idx++) {
            normA2 += a[idx] * a[idx];
            double diff = a[idx] - recon[idx];
            diff2 += diff * diff;
        }
        return Math.sqrt(diff2 / normA2);
    }

    private static double belowSubdiagonalMax(Matrix hMatrix) {
        int n = hMatrix.getRowCount();
        double max = 0.0;
        for (int i = 2; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                max = Math.max(max, Math.abs(hMatrix.get(i, j)));
            }
        }
        return max;
    }
}
