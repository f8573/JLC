package net.faulj.nativeblas;

import net.faulj.compute.DispatchPolicy;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class NativeGemmIntegrationTest {

    @After
    public void cleanup() {
        System.clearProperty("jlc.backend");
        BackendRegistry.resetForTests();
    }

    @Test
    public void nativeLibraryExecutesRealGemmWhenConfigured() {
        Assume.assumeTrue("Native library path not configured", System.getProperty("jlc.native.lib.path") != null);
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();

        BackendSnapshot before = BackendRegistry.snapshot();
        assertEquals("native", before.activeBackend());
        assertEquals(NativeStatus.READY, before.nativeContext().getStatus());
        assertFalse(before.fallbackToJava());
        assertTrue(before.nativeContext().isAvailable());
        assertFalse(before.nativeContext().getWorkspaceHandle().isNull());
        assertTrue(before.nativeContext().getProviderDescription() != null
            && !before.nativeContext().getProviderDescription().isBlank());

        Matrix a = Matrix.wrap(new double[] {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        }, 2, 3);
        Matrix b = Matrix.wrap(new double[] {
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0
        }, 3, 2);
        Matrix c = Matrix.wrap(new double[] {
            1.0, 1.0,
            1.0, 1.0
        }, 2, 2);

        Gemm.gemm(a, b, c, 1.5, 0.5, DispatchPolicy.builder().enableParallel(false).parallelism(1).build());

        assertArrayEquals(new double[] {
            87.5, 96.5,
            209.0, 231.5
        }, c.getRawData(), 1e-9);
    }
}
