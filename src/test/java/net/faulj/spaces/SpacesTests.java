package net.faulj.spaces;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixAccuracyValidator;
import net.faulj.vector.Vector;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.*;

/**
 * Comprehensive test suite for all spaces-related functionality including
 * ChangeOfBasis, DirectSum, OrthogonalComplement, and SubspaceBasis.
 */
public class SpacesTests {

    private static final double EPSILON = 1e-12;
    private static final double TOLERANCE = 1e-12;
    private static final Random RNG = new Random(42);

    // ========== Utility Methods ==========

    private static Matrix randomInvertibleMatrix(int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
            a[i][i] += n;
        }
        return fromRowMajor(a);
    }

    private static Matrix randomOrthonormalBasis(int n, int k, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        Matrix M = fromRowMajor(a);
        Vector[] ortho = new Vector[k];
        for (int j = 0; j < k; j++) {
            Vector v = new Vector(M.getColumn(j));
            for (int i = 0; i < j; i++) {
                double proj = v.dot(ortho[i]);
                v = v.subtract(ortho[i].multiplyScalar(proj));
            }
            double norm = v.norm2();
            if (norm > 1e-10) {
                ortho[j] = v.multiplyScalar(1.0 / norm);
            } else {
                double[] data = new double[n];
                data[j % n] = 1.0;
                ortho[j] = new Vector(data);
            }
        }
        return new Matrix(ortho);
    }

    private static Matrix randomMatrix(int m, int n, long seed) {
        Random rnd = new Random(seed);
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 2 - 1;
            }
        }
        return fromRowMajor(a);
    }

    private static Vector randomVector(int n, long seed) {
        Random rnd = new Random(seed);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = rnd.nextDouble() * 10 - 5;
        }
        return new Vector(data);
    }

    private static List<Vector> randomVectors(int n, int count, long seed) {
        Random rnd = new Random(seed);
        List<Vector> vectors = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double[] data = new double[n];
            for (int j = 0; j < n; j++) {
                data[j] = rnd.nextDouble() * 2 - 1;
            }
            vectors.add(new Vector(data));
        }
        return vectors;
    }

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

    private static Vector vec(double... values) {
        return new Vector(values);
    }

    private void assertVectorEquals(Vector expected, Vector actual, double tolerance) {
        assertEquals(expected.dimension(), actual.dimension());
        for (int i = 0; i < expected.dimension(); i++) {
            assertEquals(expected.get(i), actual.get(i), tolerance);
        }
    }

    private void assertMatrixEquals(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int i = 0; i < expected.getRowCount(); i++) {
            for (int j = 0; j < expected.getColumnCount(); j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), tolerance);
            }
        }
    }

    // ========== ChangeOfBasis Tests ==========

    @Test
    public void testChangeOfBasisConstructorWithValidMatrix() {
        Matrix P = Matrix.Identity(3);
        ChangeOfBasis cob = new ChangeOfBasis(P);
        assertNotNull(cob);
        assertMatrixEquals(P, cob.matrix(), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testChangeOfBasisConstructorRejectsNull() {
        new ChangeOfBasis(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testChangeOfBasisConstructorRejectsNonSquareMatrix() {
        Matrix P = new Matrix(new Vector[]{
            vec(1, 2),
            vec(3, 4),
            vec(5, 6)
        });
        new ChangeOfBasis(P);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testChangeOfBasisConstructorRejectsSingularMatrix() {
        Matrix P = new Matrix(new Vector[]{
            vec(1, 0, 0),
            vec(0, 1, 0),
            vec(1, 1, 0)
        });
        new ChangeOfBasis(P);
    }

    @Test
    public void testChangeOfBasisIdentityTransformation() {
        Matrix I = Matrix.Identity(3);
        ChangeOfBasis identity = new ChangeOfBasis(I);
        Vector x = randomVector(3, 100);
        Vector result = identity.toC(x);
        assertVectorEquals(x, result, EPSILON);
    }

    @Test
    public void testChangeOfBasisRoundTrip() {
        Matrix P = randomInvertibleMatrix(4, 101);
        ChangeOfBasis identity = new ChangeOfBasis(P);
        Vector original = randomVector(4, 101);
        Vector toC = identity.toC(original);
        Vector back = identity.toB(toC);
        assertVectorEquals(original, back, EPSILON);
    }

    @Test
    public void testChangeOfBasisSimpleBasisChange2D() {
        Matrix P = new Matrix(new Vector[]{
            vec(0, 1),
            vec(-1, 0)
        });
        ChangeOfBasis cob = new ChangeOfBasis(P);
        Vector xB = vec(1, 0);
        Vector xStd = cob.toC(xB);
        assertVectorEquals(vec(0, 1), xStd, EPSILON);
        Vector yB = vec(0, 1);
        Vector yStd = cob.toC(yB);
        assertVectorEquals(vec(-1, 0), yStd, EPSILON);
    }

    @Test
    public void testChangeOfBasisInverse() {
        Matrix P = randomInvertibleMatrix(5, 600);
        ChangeOfBasis cob = new ChangeOfBasis(P);
        ChangeOfBasis cobInv = cob.inverse();
        Vector xB = randomVector(5, 601);
        Vector xC = cob.toC(xB);
        Vector backToB = cobInv.toC(xC);
        assertVectorEquals(xB, backToB, EPSILON);
    }

    @Test
    public void testChangeOfBasisComposition() {
        Matrix P1 = randomInvertibleMatrix(4, 900);
        Matrix P2 = randomInvertibleMatrix(4, 901);
        ChangeOfBasis cobBC = new ChangeOfBasis(P1);
        ChangeOfBasis cobCD = new ChangeOfBasis(P2);
        ChangeOfBasis cobBD = cobCD.compose(cobBC);
        Vector xB = randomVector(4, 902);
        Vector xC = cobBC.toC(xB);
        Vector xD_manual = cobCD.toC(xC);
        Vector xD_direct = cobBD.toC(xB);
        assertVectorEquals(xD_manual, xD_direct, EPSILON);
    }

    // ========== DirectSum Tests ==========

    @Test
    public void testDirectSumConstructorWith2DSubspaces() {
        Matrix U = new Matrix(new Vector[]{vec(1, 0)});
        Matrix W = new Matrix(new Vector[]{vec(0, 1)});
        DirectSum ds = new DirectSum(U, W);
        assertEquals(2, ds.ambientDimension());
        assertEquals(1, ds.dimU());
        assertEquals(1, ds.dimW());
        assertEquals(2, ds.dimV());
        assertTrue(ds.isSquareInvertible());
    }

    @Test
    public void testDirectSumConstructorWith3DSubspaces() {
        Matrix U = new Matrix(new Vector[]{
            vec(1, 0, 0),
            vec(0, 1, 0)
        });
        Matrix W = new Matrix(new Vector[]{
            vec(0, 0, 1)
        });
        DirectSum ds = new DirectSum(U, W);
        assertEquals(3, ds.ambientDimension());
        assertEquals(2, ds.dimU());
        assertEquals(1, ds.dimW());
        assertEquals(3, ds.dimV());
        assertTrue(ds.isSquareInvertible());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDirectSumConstructorRejectsNullBases() {
        Matrix U = new Matrix(new Vector[]{vec(1, 0)});
        new DirectSum(null, U);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDirectSumConstructorRejectsDimensionMismatch() {
        Matrix U = new Matrix(new Vector[]{vec(1, 0)});
        Matrix W = new Matrix(new Vector[]{vec(1, 0, 0)});
        new DirectSum(U, W);
    }

    @Test
    public void testDirectSumDecomposition2D() {
        Matrix U = new Matrix(new Vector[]{vec(1, 0)});
        Matrix W = new Matrix(new Vector[]{vec(0, 1)});
        DirectSum ds = new DirectSum(U, W);
        Vector v = vec(3, 4);
        DirectSum.Decomposition decomp = ds.decompose(v);
        assertVectorEquals(vec(3, 0), decomp.u(), EPSILON);
        assertVectorEquals(vec(0, 4), decomp.w(), EPSILON);
        assertTrue(decomp.residual() < TOLERANCE);
    }

    @Test
    public void testDirectSumProjectionMatricesSumToIdentity() {
        Matrix U = new Matrix(new Vector[]{
            vec(1, 0, 0),
            vec(0, 1, 0)
        });
        Matrix W = new Matrix(new Vector[]{
            vec(0, 0, 1)
        });
        DirectSum ds = new DirectSum(U, W);
        Matrix PU = ds.projectionMatrixOntoUAlongW();
        Matrix PW = ds.projectionMatrixOntoWAlongU();
        Matrix sum = PU.add(PW);
        Matrix identity = Matrix.Identity(3);
        MatrixAccuracyValidator.ValidationResult result = 
            MatrixAccuracyValidator.validate(identity, sum, "Projections Sum to Identity");
        assertTrue(result.passes);
    }

    // ========== OrthogonalComplement Tests ==========

    @Test
    public void testOrthogonalComplementFactoryWithSingleVector() {
        List<Vector> W = List.of(vec(1, 0, 0));
        OrthogonalComplement oc = OrthogonalComplement.of(W);
        assertEquals(3, oc.ambientDimension());
        assertEquals(1, oc.dimensionW());
        assertEquals(2, oc.dimensionPerp());
    }

    @Test
    public void testOrthogonalComplementFactoryWithMultipleVectors() {
        List<Vector> W = List.of(
            vec(1, 0, 0),
            vec(0, 1, 0)
        );
        OrthogonalComplement oc = OrthogonalComplement.of(W);
        assertEquals(3, oc.ambientDimension());
        assertEquals(2, oc.dimensionW());
        assertEquals(1, oc.dimensionPerp());
    }

    @Test(expected = NullPointerException.class)
    public void testOrthogonalComplementFactoryRejectsNull() {
        OrthogonalComplement.of(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOrthogonalComplementFactoryRejectsEmptyList() {
        List<Vector> empty = List.of();
        OrthogonalComplement.of(empty);
    }

    @Test
    public void testOrthogonalComplementBasisOrthogonality() {
        List<Vector> W = randomVectors(5, 2, 100);
        OrthogonalComplement oc = OrthogonalComplement.of(W);
        List<Vector> basisW = oc.basisW();
        List<Vector> basisPerp = oc.basisPerp();
        for (Vector w : basisW) {
            for (Vector p : basisPerp) {
                double dot = w.dot(p);
                assertEquals(0.0, dot, EPSILON);
            }
        }
    }

    @Test
    public void testOrthogonalComplementProjection() {
        List<Vector> W = List.of(
            vec(1, 0, 0),
            vec(0, 1, 0)
        );
        OrthogonalComplement oc = OrthogonalComplement.of(W);
        Vector v = vec(2, 3, 5);
        Vector projPerp = oc.projectOntoPerp(v);
        assertVectorEquals(vec(0, 0, 5), projPerp, EPSILON);
    }

    @Test
    public void testOrthogonalComplementProjectionSumEqualsOriginal() {
        List<Vector> W = randomVectors(5, 2, 500);
        OrthogonalComplement oc = OrthogonalComplement.of(W);
        Vector v = randomVector(5, 501);
        Vector projW = oc.projectOntoW(v);
        Vector projPerp = oc.projectOntoPerp(v);
        Vector sum = projW.add(projPerp);
        assertVectorEquals(v, sum, EPSILON);
    }

    // ========== SubspaceBasis Tests ==========

    @Test(expected = NullPointerException.class)
    public void testSubspaceBasisRowSpaceBasisRejectsNull() {
        SubspaceBasis.rowSpaceBasis(null);
    }

    @Test
    public void testSubspaceBasisRowSpaceBasisWithZeroMatrix() {
        Matrix Z = Matrix.zero(3, 3);
        Set<Vector> basis = SubspaceBasis.rowSpaceBasis(Z);
        assertEquals(0, basis.size());
    }

    @Test
    public void testSubspaceBasisRowSpaceBasisWithIdentity() {
        Matrix I = Matrix.Identity(4);
        Set<Vector> basis = SubspaceBasis.rowSpaceBasis(I);
        assertEquals(4, basis.size());
    }

    @Test
    public void testSubspaceBasisRowSpaceBasisOrthonormality() {
        Matrix A = randomMatrix(4, 5, 100);
        Set<Vector> basis = SubspaceBasis.rowSpaceBasis(A);
        for (Vector q : basis) {
            assertEquals(1.0, q.norm2(), EPSILON);
        }
        Vector[] basisArray = basis.toArray(new Vector[0]);
        for (int i = 0; i < basisArray.length; i++) {
            for (int j = i + 1; j < basisArray.length; j++) {
                double dot = basisArray[i].dot(basisArray[j]);
                assertEquals(0.0, dot, EPSILON);
            }
        }
    }

    @Test
    public void testSubspaceBasisRankDeficientMatrix() {
        Matrix A = new Matrix(new double[][]{
            {1, 2, 3},
            {2, 4, 6},
            {3, 6, 9}
        });
        Set<Vector> basis = SubspaceBasis.rowSpaceBasis(A);
        assertEquals(1, basis.size());
    }

    @Test(expected = NullPointerException.class)
    public void testSubspaceBasisColumnSpaceBasisRejectsNull() {
        SubspaceBasis.columnSpaceBasis(null);
    }

    @Test
    public void testSubspaceBasisColumnSpaceBasisWithIdentity() {
        Matrix I = Matrix.Identity(5);
        Set<Vector> basis = SubspaceBasis.columnSpaceBasis(I);
        assertEquals(5, basis.size());
    }

    @Test(expected = NullPointerException.class)
    public void testSubspaceBasisNullSpaceBasisRejectsNull() {
        SubspaceBasis.nullSpaceBasis(null);
    }

    @Test
    public void testSubspaceBasisNullSpaceBasisWithIdentity() {
        Matrix I = Matrix.Identity(4);
        Set<Vector> basis = SubspaceBasis.nullSpaceBasis(I);
        assertNotNull(basis);
        assertEquals(0, basis.size());
    }

    @Test
    public void testSubspaceBasisNullSpaceBasisVectorsAreInNullSpace() {
        Matrix A = randomMatrix(3, 5, 500);
        Set<Vector> nullBasis = SubspaceBasis.nullSpaceBasis(A);
        assertNotNull(nullBasis);
        for (Vector v : nullBasis) {
            Matrix vMat = v.toMatrix();
            Matrix Av = A.multiply(vMat);
            for (int i = 0; i < Av.getRowCount(); i++) {
                assertEquals(0.0, Av.get(i, 0), EPSILON);
            }
        }
    }

    @Test
    public void testSubspaceBasisRankNullityTheorem() {
        Matrix A = randomMatrix(3, 6, 700);
        Set<Vector> rowBasis = SubspaceBasis.rowSpaceBasis(A);
        Set<Vector> nullBasis = SubspaceBasis.nullSpaceBasis(A);
        assertNotNull(rowBasis);
        assertNotNull(nullBasis);
        int n = 6;
        assertEquals(n, rowBasis.size() + nullBasis.size());
    }

    @Test
    public void testSubspaceBasisRankOneMatrix() {
        Matrix A = new Matrix(new double[][]{
            {1, 2, 3},
            {2, 4, 6},
            {3, 6, 9}
        });
        Set<Vector> rowBasis = SubspaceBasis.rowSpaceBasis(A);
        Set<Vector> colBasis = SubspaceBasis.columnSpaceBasis(A);
        Set<Vector> nullBasis = SubspaceBasis.nullSpaceBasis(A);
        assertEquals(1, rowBasis.size());
        assertEquals(1, colBasis.size());
        assertEquals(2, nullBasis.size());
    }
}
