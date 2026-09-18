package net.faulj.decomposition.hessenberg;

import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.eigen.qr.BlockedHessenbergQR;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixUtils;
import org.junit.After;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class HessenbergCompatibilityTest {

    @After
    public void cleanup() {
        System.clearProperty("net.faulj.decomposition.hessenberg.blockSize");
        System.clearProperty("net.faulj.eigen.qr.BlockedHessenbergQR.blockSize");
        System.clearProperty("net.faulj.eigen.qr.blockSize");
    }

    @Test
    public void blockedFacadeMatchesCanonicalReduction() {
        BlockedHessenbergQR.setBlockSize(16);
        Matrix a = randomMatrix(48, 91L);

        HessenbergResult canonical = HessenbergReduction.decompose(a);
        HessenbergResult facade = BlockedHessenbergQR.decompose(a);

        double hError = MatrixUtils.relativeError(canonical.getH(), facade.getH());
        double qError = MatrixUtils.relativeError(canonical.getQ(), facade.getQ());

        assertTrue("Facade H diverged from canonical path: " + hError, hError < 1e-12);
        assertTrue("Facade Q diverged from canonical path: " + qError, qError < 1e-12);
        assertTrue("Canonical residual too large: " + canonical.residualNorm(), canonical.residualNorm() < 1e-10);
        assertTrue("Facade residual too large: " + facade.residualNorm(), facade.residualNorm() < 1e-10);
        assertTrue("Canonical Q lost orthogonality: " + MatrixUtils.orthogonalityError(canonical.getQ()),
            MatrixUtils.orthogonalityError(canonical.getQ()) < 1e-10);
        assertTrue("Facade Q lost orthogonality: " + MatrixUtils.orthogonalityError(facade.getQ()),
            MatrixUtils.orthogonalityError(facade.getQ()) < 1e-10);
    }

    private static Matrix randomMatrix(int n, long seed) {
        Random random = new Random(seed);
        double[] data = new double[n * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return Matrix.wrap(data, n, n);
    }
}
