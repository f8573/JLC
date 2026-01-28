package net.faulj.stress;

import net.faulj.compute.DispatchPolicy;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.decomposition.result.HessenbergResult;
import net.faulj.decomposition.result.QRResult;
import net.faulj.decomposition.result.SchurResult;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.matrix.Matrix;
import net.faulj.scalar.Complex;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Stress tests decomposition accuracy across many random trials and sizes.
 */
public class StressAccuracyTest {

    private static final int TRIALS = 2000;
    private static final int MIN_SIZE = 2;
    private static final int MAX_SIZE = 8;
    private static final int[] LARGE_SIZES = {50, 100, 200, 1000};
    private static final double QR_RESIDUAL_LIMIT = 1e-2;
    private static final double HESS_RESIDUAL_LIMIT = 1e-2;
    private static final double SCHUR_RESIDUAL_LIMIT = 5e-2;
    private static final double LARGE_QR_RESIDUAL_LIMIT = 5e-2;
    private static final double LARGE_HESS_RESIDUAL_LIMIT = 5e-2;
    private static final double TRACE_TOL = 1e-5;
    // How often to print progress during the main stress loop (in trials)
    private static final int PROGRESS_INTERVAL = 100;

    /**
     * Runs repeated QR/Hessenberg/Schur decompositions and checks residuals.
     */
    @Test
    public void stressDecompositionAndEigenAccuracy() {
        Random rnd = new Random(1234567L);
        for (int t = 0; t < TRIALS; t++) {
            int n = MIN_SIZE + (t % (MAX_SIZE - MIN_SIZE + 1));
            Matrix A = randomMatrix(rnd, n);
            QRResult qr = HouseholderQR.decompose(A);
            double qrRes = qr.residualNorm();
            assertTrue("QR residual too large at size " + n + " trial " + t + ": " + qrRes,
                qrRes < QR_RESIDUAL_LIMIT);

            HessenbergResult hess = HessenbergReduction.decompose(A);
            double hessRes = hess.residualNorm();
            assertTrue("Hessenberg residual too large at size " + n + " trial " + t + ": " + hessRes,
                hessRes < HESS_RESIDUAL_LIMIT);

            SchurResult schur = RealSchurDecomposition.decompose(A);
            double schurRes = schur.residualNorm();
            assertTrue("Schur residual too large at size " + n + " trial " + t + ": " + schurRes,
                schurRes < SCHUR_RESIDUAL_LIMIT);

            if (t % PROGRESS_INTERVAL == 0) {
            System.out.printf("[stress] trial=%d size=%d QR=%.3e HESS=%.3e SCHUR=%.3e trace=%.6f%n",
                t, n, qrRes, hessRes, schurRes, A.trace());
            }

            Complex[] eigenvalues = schur.getEigenvalues();
            double sumReal = 0.0;
            double sumImag = 0.0;
            for (Complex eigenvalue : eigenvalues) {
                sumReal += eigenvalue.real;
                sumImag += eigenvalue.imag;
            }
                if (t % (PROGRESS_INTERVAL * 5) == 0) {
                System.out.printf("[eigs] trial=%d size=%d sumReal=%.6f sumImag=%.6f%n",
                    t, n, sumReal, sumImag);
                }

                assertEquals("Trace mismatch at size " + n + " trial " + t,
                    A.trace(), sumReal, TRACE_TOL);
                assertEquals("Imaginary sum should be near zero at size " + n + " trial " + t,
                    0.0, sumImag, TRACE_TOL);
        }
    }

    /**
     * Benchmarks larger matrices with size-based dispatch policy settings.
     */
    @Test
    public void stressLargeMatrixDecompositions() {
        Random rnd = new Random(7654321L);

        // System information
        System.out.printf("[large] Processor count: %d%n", Runtime.getRuntime().availableProcessors());
        System.out.printf("[large] CUDA system property: %s%n", System.getProperty("faulj.cuda.enabled", "not set"));

        for (int size : LARGE_SIZES) {
            System.out.printf("[large] starting size=%d%n", size);
            
            // Configure dispatch policy based on matrix size
            DispatchPolicy policy = configurePolicy(size);
            DispatchPolicy.setGlobalPolicy(policy);
            DispatchPolicy.Algorithm algorithm = policy.selectForMultiply(size, size, size);
            String algoLabel = algorithmLabel(algorithm);
            
            System.out.printf("[large] size=%d using: parallel=%s, cuda=%s, blockThreshold=%d, parallelThreshold=%d%n",
                size, policy.isParallelEnabled(), policy.isCudaEnabled(), 
                policy.getBlockedThreshold(), policy.getParallelThreshold());
            
            Matrix A = randomMatrix(rnd, size);

            // QR decomposition with timing
            long qrStart = System.nanoTime();
            QRResult qr = HouseholderQR.decompose(A);
            long qrTime = System.nanoTime() - qrStart;
            double qrRes = qr.residualNorm();
            System.out.printf("[large] Performed %s QR on %dx%d matrix in %.3fs (residual=%.3e, throughput=%.2fMFlops/s)%n", 
                algoLabel, size, size, qrTime / 1e9, qrRes, computeThroughput(size, qrTime));
            assertTrue("QR residual too large at size " + size + ": " + qrRes,
                qrRes < LARGE_QR_RESIDUAL_LIMIT);

            // Hessenberg reduction with timing
            long hessStart = System.nanoTime();
            HessenbergResult hess = HessenbergReduction.decompose(A);
            long hessTime = System.nanoTime() - hessStart;
            double hessRes = hess.residualNorm();
            System.out.printf("[large] Performed %s HESS on %dx%d matrix in %.3fs (residual=%.3e, throughput=%.2fMFlops/s)%n", 
                algoLabel, size, size, hessTime / 1e9, hessRes, computeThroughput(size, hessTime));
            assertTrue("Hessenberg residual too large at size " + size + ": " + hessRes,
                hessRes < LARGE_HESS_RESIDUAL_LIMIT);
        }
        
        // Reset to default policy
        DispatchPolicy.resetGlobalPolicy();
    }

    /**
     * Configure DispatchPolicy based on matrix size and hardware capabilities.
     * 
     * Size-based strategy:
     * - < 100: Standard multiply, no parallelism
     * - 100-200: Enable parallelism with lower thresholds
     * - 200-500: Blocked multiply with parallel disabled
     * - >= 500: CUDA if available, otherwise BLAS3 kernels
     */
    private static DispatchPolicy configurePolicy(int size) {
        DispatchPolicy.Builder builder = DispatchPolicy.builder();
        
        if (size < 100) {
            // Small matrices: use standard multiply, disable parallelism overhead
            builder.naiveThreshold(128)
                   .parallelThreshold(Integer.MAX_VALUE)
                   .enableParallel(false)
                   .enableCuda(false)
                   .enableBlas3(false);
        } else if (size < 1000) {
            // Medium matrices: enable parallelism with lower threshold
            builder.blockedThreshold(64)
                   .parallelThreshold(100)
                   .enableParallel(true)
                   .parallelism(Runtime.getRuntime().availableProcessors())
                   .enableCuda(false)
                   .enableBlas3(false);
        } else if (size < 3000) {
            // Large matrices: blocked multiply without parallelism overhead
            builder.blockedThreshold(64)
                   .parallelThreshold(Integer.MAX_VALUE)
                   .parallelism(Runtime.getRuntime().availableProcessors())
                   .enableParallel(true)
                   .enableCuda(false)
                   .enableBlas3(true)
                   .blas3Threshold(256);
        } else {
            // Very large matrices: CUDA if available, otherwise BLAS3
            builder.blockedThreshold(64)
                   .parallelThreshold(Integer.MAX_VALUE)
                   .enableParallel(true)
                   .parallelism(Runtime.getRuntime().availableProcessors())
                   .enableCuda(true)
                   .cudaMinDim(300)
                   .cudaMinElements(500_000L)
                   .enableBlas3(true)
                   .blas3Threshold(256);
        }
        
        return builder.build();
    }

    /**
     * Formats a dispatch algorithm label for logging.
     *
     * @param algorithm chosen algorithm
     * @return label for output
     */
    private static String algorithmLabel(DispatchPolicy.Algorithm algorithm) {
        if (algorithm == null) {
            return "STANDARD";
        }
        switch (algorithm) {
            case CUDA:
                return "CUDA";
            case PARALLEL:
                return "PARALLEL";
            case BLOCKED:
                return "BLOCK";
            case BLAS3:
                return "BLAS3";
            case STRASSEN:
                return "STRASSEN";
            case NAIVE:
            case SPECIALIZED:
            default:
                return "STANDARD";
        }
    }

    /**
     * Compute throughput in MFlops/s.
     * QR decomposition: ~(4/3)mn^2 - (2/3)n^3 flops ≈ (2/3)n^3 for square matrices
     * Hessenberg reduction: ~(10/3)n^3 flops
     */
    private static double computeThroughput(int n, long nanos) {
        if (nanos == 0) return 0.0;
        // Conservative estimate: 2n^3 flops for decomposition operations
        double flops = 2.0 * n * n * n;
        return (flops / 1e6) / (nanos / 1e9);
    }

    /**
     * Generates a random $n\times n$ matrix with entries in $[-1, 1]$.
     *
     * @param rnd random source
     * @param n matrix dimension
     * @return random matrix
     */
    private static Matrix randomMatrix(Random rnd, int n) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = rnd.nextDouble() * 2.0 - 1.0;
            }
        }
        return new Matrix(data);
    }
}
