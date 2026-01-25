package net.faulj.condition;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.*;

public class ConditionNumberTest {

    @Test
    public void testComputeNorm2SVD_diag() {
        double[][] data = new double[][]{
                {3.0, 0.0, 0.0},
                {0.0, 2.0, 0.0},
                {0.0, 0.0, 1.0}
        };

        Matrix A = new Matrix(data);
        double kappa = ConditionNumber.computeNorm2SVD(A);

        assertEquals(3.0, kappa, 1e-12);
    }
}
