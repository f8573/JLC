package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NativeOffHeapGemmIntegrationTest {

    @After
    public void cleanup() {
        System.clearProperty("jlc.backend");
        BackendRegistry.resetForTests();
    }

    @Test
    public void nativeBackendExecutesOffHeapRowMajorGemm() {
        assumeNativeBackendReady();

        try (OffHeapMatrix a = new OffHeapMatrix(2, 3);
             OffHeapMatrix b = new OffHeapMatrix(3, 2);
             OffHeapMatrix c = new OffHeapMatrix(2, 2)) {
            fill(a, new double[] {
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0
            });
            fill(b, new double[] {
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0
            });
            fill(c, new double[] {
                1.0, 1.0,
                1.0, 1.0
            });

            Gemm.gemm(a, b, c, 1.5, 0.5, DispatchPolicy.builder().enableParallel(false).parallelism(1).build());

            assertEquals(87.5, c.get(0, 0), 1e-9);
            assertEquals(96.5, c.get(0, 1), 1e-9);
            assertEquals(209.0, c.get(1, 0), 1e-9);
            assertEquals(231.5, c.get(1, 1), 1e-9);
        }
    }

    @Test
    public void nativeBackendExecutesOffHeapColMajorInputGemm() {
        assumeNativeBackendReady();

        try (OffHeapMatrix a = OffHeapMatrix.allocate(2, 3, OffHeapMatrix.Order.COL_MAJOR, OffHeapMatrix.DEFAULT_ALIGNMENT);
             OffHeapMatrix b = new OffHeapMatrix(3, 2);
             OffHeapMatrix c = new OffHeapMatrix(2, 2)) {
            fill(a, new double[] {
                2.0, 1.0, 0.0,
                -1.0, 3.0, 4.0
            });
            fill(b, new double[] {
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0
            });

            Gemm.gemm(a, b, c, 1.0, 0.0, DispatchPolicy.builder().enableParallel(false).parallelism(1).build());

            assertEquals(5.0, c.get(0, 0), 1e-9);
            assertEquals(8.0, c.get(0, 1), 1e-9);
            assertEquals(28.0, c.get(1, 0), 1e-9);
            assertEquals(34.0, c.get(1, 1), 1e-9);
        }
    }

    @Test
    public void nativeBackendExecutesOffHeapColMajorOutputGemm() {
        assumeNativeBackendReady();

        try (OffHeapMatrix a = new OffHeapMatrix(2, 3);
             OffHeapMatrix b = new OffHeapMatrix(3, 2);
             OffHeapMatrix c = OffHeapMatrix.allocate(2, 2, OffHeapMatrix.Order.COL_MAJOR, OffHeapMatrix.DEFAULT_ALIGNMENT)) {
            fill(a, new double[] {
                1.0, -2.0, 3.0,
                0.5, 4.0, -1.0
            });
            fill(b, new double[] {
                2.0, 1.0,
                -3.0, 0.5,
                4.0, -2.0
            });
            fill(c, new double[] {
                10.0, 20.0,
                30.0, 40.0
            });

            Gemm.gemm(a, b, c, 2.0, 0.25, DispatchPolicy.builder().enableParallel(false).parallelism(1).build());

            assertEquals(42.5, c.get(0, 0), 1e-9);
            assertEquals(-7.0, c.get(0, 1), 1e-9);
            assertEquals(-22.5, c.get(1, 0), 1e-9);
            assertEquals(19.0, c.get(1, 1), 1e-9);
        }
    }

    @Test
    public void nativeBackendHandlesMixedHeapAndOffHeapOperands() {
        assumeNativeBackendReady();

        Matrix a = Matrix.wrap(new double[] {
            1.0, 2.0, 3.0,
            -1.0, 0.0, 4.0
        }, 2, 3);
        try (OffHeapMatrix b = new OffHeapMatrix(3, 2);
             OffHeapMatrix c = new OffHeapMatrix(2, 2)) {
            fill(b, new double[] {
                2.0, 5.0,
                3.0, 7.0,
                11.0, 13.0
            });
            fill(c, new double[] {
                1.0, 2.0,
                3.0, 4.0
            });

            Gemm.gemm(a, b, c, 1.0, 1.0, DispatchPolicy.builder().enableParallel(false).parallelism(1).build());

            assertEquals(42.0, c.get(0, 0), 1e-9);
            assertEquals(60.0, c.get(0, 1), 1e-9);
            assertEquals(45.0, c.get(1, 0), 1e-9);
            assertEquals(51.0, c.get(1, 1), 1e-9);
        }
    }

    private static void assumeNativeBackendReady() {
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        Assume.assumeTrue("Native backend unavailable: " + snapshot.nativeContext().getMessage(),
            "native".equals(snapshot.activeBackend()));
    }

    private static void fill(OffHeapMatrix matrix, double[] values) {
        int index = 0;
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                matrix.set(row, col, values[index++]);
            }
        }
    }
}
