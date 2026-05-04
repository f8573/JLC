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

    @Test
    public void nativeBackendUsesAvx2PackedMicrokernelOnWindowsBuild() {
        Assume.assumeTrue("Native library path not configured", System.getProperty("jlc.native.lib.path") != null);
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();

        BackendSnapshot snapshot = BackendRegistry.snapshot();
        assertEquals("native", snapshot.activeBackend());
        assertEquals("builtin only", snapshot.nativeContext().getProviderDescription());
        assertTrue("Expected AVX2 runtime, got: " + snapshot.nativeContext().getRuntimeDescription(),
            snapshot.nativeContext().getRuntimeDescription() != null
                && snapshot.nativeContext().getRuntimeDescription().contains("AVX2"));

        NativeProfiling.setEnabled(true);
        NativeProfiling.reset();
        double[] a = sequence(10 * 16, 0.01);
        double[] b = sequence(16 * 8, -0.02);
        double[] c = new double[10 * 8];
        NativeBindings.nativeGemm(a, 10, 16, b, 16, 8, c, 10, 8, 1.0, 0.0, 1, NativeFlags.FORCE_BUILTIN);
        NativeGemmProfile profile = NativeProfiling.snapshot().orElse(NativeGemmProfile.EMPTY);
        NativeProfiling.setEnabled(false);

        assertEquals("AVX2 MR regression", 5L, profile.lastMr());
        assertEquals("AVX2 NR regression", 4L, profile.lastNr());
        assertTrue("Expected packed microkernel calls", profile.microtileCalls() > 0);
    }

    private static double[] sequence(int length, double scale) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = scale * (i + 1);
        }
        return out;
    }
}
