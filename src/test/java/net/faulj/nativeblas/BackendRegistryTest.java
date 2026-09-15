package net.faulj.nativeblas;

import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class BackendRegistryTest {

    @After
    public void cleanup() {
        System.clearProperty("jlc.backend");
        System.clearProperty("faulj.backend");
        BackendRegistry.resetForTests();
    }

    @Test
    public void javaBackendIsDefaultWhenRequestedExplicitly() {
        System.setProperty("jlc.backend", "java");
        BackendRegistry.resetForTests();

        BackendSnapshot snapshot = BackendRegistry.snapshot();

        assertEquals(BackendMode.JAVA, snapshot.requestedBackend());
        assertEquals("java", snapshot.activeBackend());
        assertFalse(snapshot.fallbackToJava());
        assertEquals(NativeStatus.NOT_REQUESTED, snapshot.nativeContext().getStatus());
    }

    @Test
    public void nativeBackendRequestFallsBackToJavaWhenLibraryIsUnavailable() {
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();

        BackendSnapshot snapshot = BackendRegistry.snapshot();

        assertEquals(BackendMode.NATIVE, snapshot.requestedBackend());
        if ("native".equals(snapshot.activeBackend())) {
            assertFalse(snapshot.fallbackToJava());
            assertEquals(NativeStatus.READY, snapshot.nativeContext().getStatus());
            assertFalse(snapshot.nativeContext().getWorkspaceHandle().isNull());
        } else {
            assertEquals("java", snapshot.activeBackend());
            assertTrue(snapshot.fallbackToJava());
            assertTrue(snapshot.nativeContext().getStatus() == NativeStatus.LOAD_FAILED
                || snapshot.nativeContext().getStatus() == NativeStatus.NOT_REQUESTED);
        }
    }

    @Test
    public void autoModeStillUsesJavaBackendWhenLibraryIsUnavailable() {
        System.setProperty("jlc.backend", "auto");
        BackendRegistry.resetForTests();

        BackendSnapshot snapshot = BackendRegistry.snapshot();

        assertEquals(BackendMode.AUTO, snapshot.requestedBackend());
        if ("native".equals(snapshot.activeBackend())) {
            assertFalse(snapshot.fallbackToJava());
            assertEquals(NativeStatus.READY, snapshot.nativeContext().getStatus());
            assertFalse(snapshot.nativeContext().getWorkspaceHandle().isNull());
        } else {
            assertEquals("java", snapshot.activeBackend());
            assertTrue(snapshot.fallbackToJava());
        }
    }

    @Test
    public void gemmProducesCorrectResultWhenNativeBackendIsRequested() {
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();

        Matrix a = Matrix.wrap(new double[] {
            1.0, 2.0,
            3.0, 4.0
        }, 2, 2);
        Matrix b = Matrix.wrap(new double[] {
            5.0, 6.0,
            7.0, 8.0
        }, 2, 2);
        Matrix c = Matrix.zero(2, 2);

        Gemm.gemm(a, b, c, 1.0, 0.0, null);

        assertArrayEquals(new double[] {19.0, 22.0, 43.0, 50.0}, c.getRawData(), 1e-12);
    }
}
