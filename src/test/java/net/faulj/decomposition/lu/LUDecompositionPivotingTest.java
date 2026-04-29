package net.faulj.decomposition.lu;

import net.faulj.decomposition.result.LUResult;
import net.faulj.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LUDecompositionPivotingTest {

    @Test
    public void partialPivotingHandlesLeadingZeroPivot() {
        Matrix a = new Matrix(new double[][]{
            {0.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 10.0}
        });

        LUResult result = new LUDecomposition().decompose(a);

        assertTrue("Expected at least one row exchange", result.getP().getExchangeCount() > 0);
        assertTrue("Residual too large: " + result.residualNorm(), result.residualNorm() < 1e-12);

        Matrix l = result.getL();
        Matrix u = result.getU();
        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, l.get(i, i), 0.0);
            for (int j = i + 1; j < 3; j++) {
                assertEquals(0.0, l.get(i, j), 1e-15);
            }
            for (int j = 0; j < i; j++) {
                assertEquals(0.0, u.get(i, j), 1e-15);
            }
        }
    }
}
