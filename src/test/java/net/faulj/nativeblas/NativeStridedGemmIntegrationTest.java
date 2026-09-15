package net.faulj.nativeblas;

import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

public class NativeStridedGemmIntegrationTest {

    @After
    public void cleanup() {
        BackendRegistry.resetForTests();
    }

    @Test
    public void nativeRowMajorStridedMatchesJavaBackend() {
        NativeBackend nativeBackend = nativeBackend();
        JavaBackend javaBackend = new JavaBackend();

        double[] a = new double[24];
        double[] b = new double[20];
        double[] cNative = new double[20];
        double[] cJava = new double[20];
        fill(a, 1.0);
        fill(b, -0.5);
        fill(cNative, 0.25);
        System.arraycopy(cNative, 0, cJava, 0, cNative.length);

        nativeBackend.gemmStrided(a, 1, 6, b, 2, 5, cNative, 3, 4, 3, 4, 2, 1.25, -0.5);
        javaBackend.gemmStrided(a, 1, 6, b, 2, 5, cJava, 3, 4, 3, 4, 2, 1.25, -0.5);

        assertArrayEquals(cJava, cNative, 1e-9);
    }

    @Test
    public void nativeTransposeAStridedMatchesJavaBackend() {
        NativeBackend nativeBackend = nativeBackend();
        JavaBackend javaBackend = new JavaBackend();

        double[] a = new double[32];
        double[] b = new double[30];
        double[] cNative = new double[18];
        double[] cJava = new double[18];
        fill(a, -1.0);
        fill(b, 0.75);
        fill(cNative, 1.0);
        System.arraycopy(cNative, 0, cJava, 0, cNative.length);

        nativeBackend.gemmStridedTransA(a, 1, 6, b, 0, 5, cNative, 0, 6, 5, 3, 3, -0.75, 0.5, 3);
        javaBackend.gemmStridedTransA(a, 1, 6, b, 0, 5, cJava, 0, 6, 5, 3, 3, -0.75, 0.5, 3);

        assertArrayEquals(cJava, cNative, 1e-9);
    }

    @Test
    public void nativeColMajorAStridedMatchesJavaBackend() {
        NativeBackend nativeBackend = nativeBackend();
        JavaBackend javaBackend = new JavaBackend();

        double[] a = new double[30];
        double[] b = new double[21];
        double[] cNative = new double[20];
        double[] cJava = new double[20];
        fill(a, 0.2);
        fill(b, -0.8);
        fill(cNative, -0.3);
        System.arraycopy(cNative, 0, cJava, 0, cNative.length);

        nativeBackend.gemmStridedColMajorA(a, 2, 6, b, 1, 5, cNative, 2, 4, 4, 3, 2, 1.0, 0.25, 3);
        javaBackend.gemmStridedColMajorA(a, 2, 6, b, 1, 5, cJava, 2, 4, 4, 3, 2, 1.0, 0.25, 3);

        assertArrayEquals(cJava, cNative, 1e-9);
    }

    @Test
    public void nativeColMajorBStridedMatchesJavaBackend() {
        NativeBackend nativeBackend = nativeBackend();
        JavaBackend javaBackend = new JavaBackend();

        double[] a = new double[28];
        double[] b = new double[30];
        double[] cNative = new double[24];
        double[] cJava = new double[24];
        fill(a, 1.5);
        fill(b, -1.5);
        fill(cNative, 0.1);
        System.arraycopy(cNative, 0, cJava, 0, cNative.length);

        nativeBackend.gemmStridedColMajorB(a, 2, 7, b, 3, 6, cNative, 1, 6, 3, 4, 3, 0.5, 1.0, 4);
        javaBackend.gemmStridedColMajorB(a, 2, 7, b, 3, 6, cJava, 1, 6, 3, 4, 3, 0.5, 1.0, 4);

        assertArrayEquals(cJava, cNative, 1e-9);
    }

    private static NativeBackend nativeBackend() {
        NativeBackend nativeBackend = new NativeBackend(new JavaBackend());
        NativeContext context = nativeBackend.probe(true);
        Assume.assumeTrue("Native backend unavailable: " + context.getMessage(), context.isAvailable());
        return nativeBackend;
    }

    private static void fill(double[] data, double seed) {
        for (int i = 0; i < data.length; i++) {
            data[i] = seed + Math.sin(i * 0.37) + (i % 5) * 0.11;
        }
    }
}
