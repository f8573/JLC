package net.faulj.determinant;

import net.faulj.matrix.Matrix;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Verifies determinant implementations against known values and cross-checks.
 */
public class DeterminantTest {

    private static final double TOL = 1e-9;

    // ========== Determinant Tests ==========

    /**
     * Checks determinant values for small matrices with known results.
     */
    @Test
    public void determinantMatchesExpected() {
        Matrix a1 = new Matrix(new double[][]{{5}});
        assertEquals(5, Determinant.compute(a1), TOL);

        Matrix a2 = new Matrix(new double[][]{{1,2},{3,4}});
        System.out.println(a2);
        assertEquals(-2, Determinant.compute(a2), TOL);

        Matrix a3 = new Matrix(new double[][]{
                {2,3,1},
                {4,5,6},
                {7,8,9}
        });
        assertEquals(9, Determinant.compute(a3), TOL);
    }

    /**
     * Confirms the facade result matches minors-based computation.
     */
    @Test
    public void determinantMatchesMinorsDeterminant() {
        Matrix a1 = new Matrix(new double[][]{{5}});
        assertEquals(MinorsDeterminant.compute(a1), Determinant.compute(a1), TOL);

        Matrix a2 = new Matrix(new double[][]{{1,2},{3,4}});
        assertEquals(MinorsDeterminant.compute(a2), Determinant.compute(a2), TOL);

        Matrix a3 = new Matrix(new double[][]{
                {2,3,1},
                {4,5,6},
                {7,8,9}
        });
        assertEquals(MinorsDeterminant.compute(a3), Determinant.compute(a3), TOL);
    }

    // ========== MinorsDeterminant Tests ==========

    /**
     * Validates minors-based determinant on small matrices.
     */
    @Test
    public void minorsDeterminantMatchesExpected() {
        Matrix a1 = new Matrix(new double[][]{{5}});
        assertEquals(5, MinorsDeterminant.compute(a1), TOL);

        Matrix a2 = new Matrix(new double[][]{{1,2},{3,4}});
        System.out.println(a2);
        assertEquals(-2, MinorsDeterminant.compute(a2), TOL);

        Matrix a3 = new Matrix(new double[][]{
                {2,3,1},
                {4,5,6},
                {7,8,9}
        });
        assertEquals(9, MinorsDeterminant.compute(a3), TOL);
    }

    // ========== LUDeterminant Tests ==========

    /**
     * Ensures LU-based determinant matches the facade for larger matrices.
     */
    @Test
    public void LUDeterminantMatchesLUResult() {
        Matrix a = new Matrix(new double[][]{
                {4,2,1,3},
                {0,5,3,1},
                {1,3,6,2},
                {3,1,2,4}
        });

        double viaLUDet = LUDeterminant.compute(a);
        double viaFacade = MinorsDeterminant.compute(a);
        // Determinant facade dispatches to LUDeterminant for n>3
        assertEquals(viaLUDet, viaFacade, TOL * Math.max(1.0, Math.abs(viaFacade)));
    }

    /**
     * Validates LU determinant value for a fixed $4\times4$ matrix.
     */
    @Test
    public void LUDeterminantValid() {
        Matrix a = new Matrix(new double[][]{
                {4,2,1,3},
                {0,5,3,1},
                {1,3,6,2},
                {3,1,2,4}
        });

        double viaLUDet = LUDeterminant.compute(a);
        // Determinant facade dispatches to LUDeterminant for n>3
        assertEquals(134, viaLUDet, TOL);
    }
}
