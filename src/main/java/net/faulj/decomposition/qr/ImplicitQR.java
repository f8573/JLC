package net.faulj.decomposition.qr;

import net.faulj.matrix.Matrix;
import net.faulj.vector.Vector;
import net.faulj.vector.VectorUtils;

public class ImplicitQR {
    public static Matrix[] decompose(Matrix Horig) {
        Matrix H = Horig.copy();

        Matrix E = H.crop(H.getRowCount()-2, H.getRowCount()-1,H.getColumnCount()-2,H.getColumnCount()-1);

        double[] eigen = new double[]{E.trace(),E.get(0, 0)*E.get(1, 1)-E.get(0, 1)*E.get(1, 0)};

        Vector x0 = H.multiply(H).subtract(H.multiplyScalar(eigen[0])).add(Matrix.Identity(H.getColumnCount()).multiplyScalar(eigen[1])).multiply(new Matrix(new Vector[]{VectorUtils.unitVector(H.getColumnCount(),0)})).getData()[0].resize(3);

        int n = H.getRowCount();
        if (n < 3) {
            System.out.println("Matrix too small for implicit QR; returning");
            return new Matrix[2];
        }

        System.out.println("Original H:");
        System.out.println(H);
        double detBefore = H.trace();
        System.out.println("trace(H) before = " + detBefore);

        java.util.ArrayList<Vector> reflectorVs = new java.util.ArrayList<>();
        java.util.ArrayList<Double> reflectorTaus = new java.util.ArrayList<>();
        java.util.ArrayList<Integer> reflectorStarts = new java.util.ArrayList<>();

        for (int k = 0; k < n - 2; k++) {
            int s;
            Vector xk;
            if (k == 0) {
                s = 0;
                Vector hh0 = VectorUtils.householder(x0.copy());
                double tau0 = hh0.get(hh0.dimension() - 1);
                Vector v0 = hh0.resize(hh0.dimension() - 1);
                double v0norm = v0.norm2();
                Vector u0 = v0.multiplyScalar(1.0 / v0norm);
                double tauNorm = 2.0;
                Matrix Q3 = Matrix.Identity(3).subtract(u0.multiply(u0.transpose()).multiplyScalar(tauNorm));

                int kblk = 3;
                Matrix A = H.crop(0, kblk - 1, 0, kblk - 1);
                Matrix B = (kblk < n) ? H.crop(0, kblk - 1, kblk, n - 1) : null;
                Matrix C = (kblk < n) ? H.crop(kblk, n - 1, 0, kblk - 1) : null;

                Matrix Anew = Q3.transpose().multiply(A).multiply(Q3);
                Matrix Bnew = (B != null) ? Q3.transpose().multiply(B) : null;
                Matrix Cnew = (C != null) ? C.multiply(Q3) : null;

                for (int i = 0; i < kblk; i++) {
                    for (int j = 0; j < kblk; j++) {
                        H.set(i, j, Anew.get(i, j));
                    }
                    if (Bnew != null) {
                        for (int j = kblk; j < n; j++) {
                            H.set(i, j, Bnew.get(i, j - kblk));
                        }
                    }
                }
                if (Cnew != null) {
                    for (int i = kblk; i < n; i++) {
                        for (int j = 0; j < kblk; j++) {
                            H.set(i, j, Cnew.get(i - kblk, j));
                        }
                    }
                }

                reflectorVs.add(u0.copy());
                reflectorTaus.add(tauNorm);
                continue;
            } else {
                s = k + 1;
                int mLen = Math.min(2, n - s);
                if (mLen <= 0) continue;
                double[] xdata = new double[mLen];
                for (int i = 0; i < mLen; i++) xdata[i] = H.get(s + i, k);
                xk = new Vector(xdata);
            }
            reflectorStarts.add(s);
            if (xk.norm2() <= 1e-10) continue;

            Vector hh = VectorUtils.householder(xk.copy());
            Vector vsmall = hh.resize(hh.dimension() - 1);
            double vnorm = vsmall.norm2();
            if (vnorm <= 1e-10) continue;
            Vector u = vsmall.multiplyScalar(1.0 / vnorm);
            double tau = 2.0;
            reflectorVs.add(u.copy());
            reflectorTaus.add(tau);

            int m = n - s;
            double[] vhatData = new double[m];
            for (int i = 0; i < vsmall.dimension(); i++) vhatData[i] = u.get(i);
            Vector vhat = new Vector(vhatData);

            Matrix Hsub = H.crop(s, n - 1, s, n - 1);

            Matrix P = Matrix.Identity(m).subtract(vhat.toMatrix().multiply(vhat.transpose()).multiplyScalar(tau));
            Matrix PP = P.multiply(P);
            double maxDiff = 0.0;
            for (int ii = 0; ii < PP.getRowCount(); ii++) {
                for (int jj = 0; jj < PP.getColumnCount(); jj++) {
                    double expect = (ii == jj) ? 1.0 : 0.0;
                    double diff = Math.abs(PP.get(ii, jj) - expect);
                    if (diff > maxDiff) maxDiff = diff;
                }
            }

            Matrix r = vhat.transpose().multiply(Hsub);
            Hsub = Hsub.subtract(vhat.toMatrix().multiply(r).multiplyScalar(tau));

            Matrix w = Hsub.multiply(vhat.toMatrix());
            Hsub = Hsub.subtract(w.multiply(vhat.transpose()).multiplyScalar(tau));

            Matrix H12 = H.crop(0, s - 1, s, n - 1);
            Matrix y = H12.multiply(vhat.toMatrix());
            H12 = H12.subtract(y.multiply(vhat.transpose()).multiplyScalar(tau));
            for (int i = 0; i < s; i++) {
                for (int j = 0; j < m; j++) {
                    H.set(i, s + j, H12.get(i, j));
                }
            }

            Matrix H21 = H.crop(s, n - 1, 0, s - 1);
            Matrix z = vhat.transpose().multiply(H21);
            H21 = H21.subtract(vhat.toMatrix().multiply(z).multiplyScalar(tau));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < s; j++) {
                    H.set(s + i, j, H21.get(i, j));
                }
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    H.set(s + i, s + j, Hsub.get(i, j));
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i > j + 1) {
                        H.set(i, j, 0.0);
                    } else {
                        double val = H.get(i, j);
                        if (Math.abs(val) < 1e-10) H.set(i, j, 0.0);
                    }
                }
            }
        }

        Matrix Qfull = Matrix.Identity(n);
        for (int idx = 0; idx < reflectorVs.size(); idx++) {
            Vector v = reflectorVs.get(idx);
            double tau = reflectorTaus.get(idx);
            int s = (idx == 0) ? 0 : idx + 1;
            int m = n - s;
            Matrix Pfull = Matrix.Identity(n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    double vi = (i < v.dimension()) ? v.get(i) : 0.0;
                    double vj = (j < v.dimension()) ? v.get(j) : 0.0;
                    double pv = ((i == j) ? 1.0 : 0.0) - tau * vi * vj;
                    Pfull.set(s + i, s + j, pv);
                }
            }
            Qfull = Qfull.multiply(Pfull);
        }

        Matrix Hnew = Qfull.transpose().multiply(Horig).multiply(Qfull);
        H = Hnew;

        System.out.println("Transformed H after bulge chase:");
        System.out.println(H);
        double detAfter = H.trace();
        System.out.println("trace(H) after  = " + detAfter);
        System.out.println("Stored " + reflectorVs.size() + " householder reflectors for reconstruction.");

        Matrix Q0 = new Matrix(reflectorVs.toArray(new Vector[0]));
        return new Matrix[]{H, Q0};
    }
}
