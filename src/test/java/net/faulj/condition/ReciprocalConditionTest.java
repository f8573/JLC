package net.faulj.condition;

import net.faulj.decomposition.lu.LUDecomposition;
import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixNorms;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Tests for reciprocal condition number estimation.
 */
public class ReciprocalConditionTest {

    /**
     * Validate reciprocal condition number for a diagonal matrix via LU.
     */
    @Test
    public void testEstimateFromLU_diagonalMatrix() {
        double[][] data = new double[][]{
                {3.0, 0.0, 0.0},
                {0.0, 2.0, 0.0},
                {0.0, 0.0, 1.0}
        };

        Matrix A = new Matrix(data);
        LUDecomposition luAlgo = new LUDecomposition();
        LUResult lu = luAlgo.decompose(A);

        double expectedNormA = MatrixNorms.norm1(A);
        double expectedNormInv = MatrixNorms.norm1(A.inverse());
        double expectedRcond = 1.0 / (expectedNormA * expectedNormInv);

        double rcond = ReciprocalCondition.estimateFromLU(lu);

        assertEquals(expectedRcond, rcond, 1e-12);
    }
}
