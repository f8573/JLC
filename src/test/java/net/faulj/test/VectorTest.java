package net.faulj.test;

import net.faulj.vector.Vector;
import org.junit.Test;
import static org.junit.Assert.*;

public class VectorTest {

    // Define a relative tolerance for all tests.
    private static final float RELATIVE_TOLERANCE = 1e-5f;

    /**
     * Test the addition of two real vectors using relative error.
     */
    @Test
    public void testVectorAddition() {
        int size = 1024;
        Vector v1 = new Vector(size);
        Vector v2 = new Vector(size);
        float[] data1 = new float[size];
        float[] data2 = new float[size];
        float[] expected = new float[size];

        for (int i = 0; i < size; i++) {
            data1[i] = i;
            data2[i] = 2 * i;
            expected[i] = i + 2 * i; // expected sum
        }

        v1.setData(data1);
        v2.setData(data2);
        Vector result = v1.add(v2);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            float error = Math.abs(expected[i] - resultData[i]);
            // For expected value of zero, use a fixed tolerance check.
            if (expected[i] == 0.0f) {
                assertTrue(
                        String.format("Vector addition failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Vector addition failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error <= RELATIVE_TOLERANCE * Math.abs(expected[i])
                );
            }
        }

        v1.free();
        v2.free();
        result.free();
    }

    /**
     * Test the subtraction of two real vectors using relative error.
     */
    @Test
    public void testVectorSubtraction() {
        int size = 1024;
        Vector v1 = new Vector(size);
        Vector v2 = new Vector(size);
        float[] data1 = new float[size];
        float[] data2 = new float[size];
        float[] expected = new float[size];

        for (int i = 0; i < size; i++) {
            data1[i] = 3 * i;
            data2[i] = i;
            expected[i] = 3 * i - i;
        }

        v1.setData(data1);
        v2.setData(data2);
        Vector result = v1.subtract(v2);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            float error = Math.abs(expected[i] - resultData[i]);
            if (expected[i] == 0.0f) {
                assertTrue(
                        String.format("Vector subtraction failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Vector subtraction failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error <= RELATIVE_TOLERANCE * Math.abs(expected[i])
                );
            }
        }

        v1.free();
        v2.free();
        result.free();
    }

    /**
     * Test scalar multiplication for a real vector using relative error.
     */
    @Test
    public void testScalarMultiplicationReal() {
        int size = 1024;
        Vector v = new Vector(size);
        float[] data = new float[size];
        float scalar = 2.5f;
        float[] expected = new float[size];

        for (int i = 0; i < size; i++) {
            data[i] = i;
            expected[i] = i * scalar;
        }

        v.setData(data);
        Vector result = v.multiply(scalar);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            float error = Math.abs(expected[i] - resultData[i]);
            if (expected[i] == 0.0f) {
                assertTrue(
                        String.format("Scalar multiplication failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Scalar multiplication failed at index %d: expected %f but got %f", i, expected[i], resultData[i]),
                        error <= RELATIVE_TOLERANCE * Math.abs(expected[i])
                );
            }
        }

        v.free();
        result.free();
    }

    /**
     * Test the dot product of two real vectors using relative error.
     */
    @Test
    public void testDotProductReal() {
        int size = 1024;
        Vector v1 = new Vector(size);
        Vector v2 = new Vector(size);
        float[] data1 = new float[size];
        float[] data2 = new float[size];
        float expectedDot = 0.0f;

        for (int i = 0; i < size; i++) {
            data1[i] = i;
            data2[i] = i;
            expectedDot += i * i;
        }

        v1.setData(data1);
        v2.setData(data2);
        float resultDot = v1.dot(v2);

        float error = Math.abs(expectedDot - resultDot);
        if (expectedDot == 0.0f) {
            assertTrue("Dot product error too high", error < RELATIVE_TOLERANCE);
        } else {
            assertTrue("Dot product relative error too high", error <= RELATIVE_TOLERANCE * Math.abs(expectedDot));
        }

        v1.free();
        v2.free();
    }

    /**
     * Test the addition of two complex vectors using relative error.
     */
    @Test
    public void testComplexAddition() {
        int size = 512;
        // Complex vectors use 2 * size values (alternating real and imaginary parts)
        Vector v1 = new Vector(size, true);
        Vector v2 = new Vector(size, true);
        float[] data1 = new float[2 * size];
        float[] data2 = new float[2 * size];
        float[] expected = new float[2 * size];

        for (int i = 0; i < size; i++) {
            // v1: (i, i)
            data1[2 * i] = i;
            data1[2 * i + 1] = i;
            // v2: (2*i, -i)
            data2[2 * i] = 2 * i;
            data2[2 * i + 1] = -i;
            // expected addition: (3*i, 0)
            expected[2 * i] = i + 2 * i;
            expected[2 * i + 1] = i + (-i);
        }

        v1.setData(data1);
        v2.setData(data2);
        Vector result = v1.add(v2);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            // Real part check
            float expReal = expected[2 * i];
            float actReal = resultData[2 * i];
            float errorReal = Math.abs(expReal - actReal);
            if (expReal == 0.0f) {
                assertTrue(
                        String.format("Complex addition failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex addition failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal <= RELATIVE_TOLERANCE * Math.abs(expReal)
                );
            }

            // Imaginary part check
            float expImag = expected[2 * i + 1];
            float actImag = resultData[2 * i + 1];
            float errorImag = Math.abs(expImag - actImag);
            if (expImag == 0.0f) {
                assertTrue(
                        String.format("Complex addition failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex addition failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag <= RELATIVE_TOLERANCE * Math.abs(expImag)
                );
            }
        }

        v1.free();
        v2.free();
        result.free();
    }

    /**
     * Test the subtraction of two complex vectors using relative error.
     */
    @Test
    public void testComplexSubtraction() {
        int size = 512;
        Vector v1 = new Vector(size, true);
        Vector v2 = new Vector(size, true);
        float[] data1 = new float[2 * size];
        float[] data2 = new float[2 * size];
        float[] expected = new float[2 * size];

        for (int i = 0; i < size; i++) {
            // v1: (3*i, 2*i)
            data1[2 * i] = 3 * i;
            data1[2 * i + 1] = 2 * i;
            // v2: (i, i)
            data2[2 * i] = i;
            data2[2 * i + 1] = i;
            // expected subtraction: (2*i, i)
            expected[2 * i] = 3 * i - i;
            expected[2 * i + 1] = 2 * i - i;
        }

        v1.setData(data1);
        v2.setData(data2);
        Vector result = v1.subtract(v2);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            // Real part check
            float expReal = expected[2 * i];
            float actReal = resultData[2 * i];
            float errorReal = Math.abs(expReal - actReal);
            if (expReal == 0.0f) {
                assertTrue(
                        String.format("Complex subtraction failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex subtraction failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal <= RELATIVE_TOLERANCE * Math.abs(expReal)
                );
            }

            // Imaginary part check
            float expImag = expected[2 * i + 1];
            float actImag = resultData[2 * i + 1];
            float errorImag = Math.abs(expImag - actImag);
            if (expImag == 0.0f) {
                assertTrue(
                        String.format("Complex subtraction failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex subtraction failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag <= RELATIVE_TOLERANCE * Math.abs(expImag)
                );
            }
        }

        v1.free();
        v2.free();
        result.free();
    }

    /**
     * Test scalar multiplication for a complex vector using relative error.
     */
    @Test
    public void testComplexScalarMultiplication() {
        int size = 512;
        Vector v = new Vector(size, true);
        float[] data = new float[2 * size];
        float scalar = 3.0f;
        float[] expected = new float[2 * size];

        for (int i = 0; i < size; i++) {
            // v: (i, -i)
            data[2 * i] = i;
            data[2 * i + 1] = -i;
            // expected: (i * scalar, -i * scalar)
            expected[2 * i] = i * scalar;
            expected[2 * i + 1] = -i * scalar;
        }

        v.setData(data);
        Vector result = v.multiply(scalar);
        float[] resultData = result.getData();

        for (int i = 0; i < size; i++) {
            // Real part check
            float expReal = expected[2 * i];
            float actReal = resultData[2 * i];
            float errorReal = Math.abs(expReal - actReal);
            if (expReal == 0.0f) {
                assertTrue(
                        String.format("Complex scalar multiplication failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex scalar multiplication failed at index %d (real): expected %f but got %f", i, expReal, actReal),
                        errorReal <= RELATIVE_TOLERANCE * Math.abs(expReal)
                );
            }

            // Imaginary part check
            float expImag = expected[2 * i + 1];
            float actImag = resultData[2 * i + 1];
            float errorImag = Math.abs(expImag - actImag);
            if (expImag == 0.0f) {
                assertTrue(
                        String.format("Complex scalar multiplication failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag < RELATIVE_TOLERANCE
                );
            } else {
                assertTrue(
                        String.format("Complex scalar multiplication failed at index %d (imaginary): expected %f but got %f", i, expImag, actImag),
                        errorImag <= RELATIVE_TOLERANCE * Math.abs(expImag)
                );
            }
        }

        v.free();
        result.free();
    }

    /**
     * Test the dot product computation for two complex vectors using relative error.
     */
    @Test
    public void testComplexDotProduct() {
        int size = 512;
        Vector v1 = new Vector(size, true);
        Vector v2 = new Vector(size, true);
        float[] data1 = new float[2 * size];
        float[] data2 = new float[2 * size];
        float expectedReal = 0.0f;
        float expectedImag = 0.0f;

        for (int i = 0; i < size; i++) {
            // v1: (i, i+1); v2: (i, -i)
            data1[2 * i] = i;
            data1[2 * i + 1] = i + 1;
            data2[2 * i] = i;
            data2[2 * i + 1] = -i;
            float a = i;
            float b = i + 1;
            float c = i;
            float d = -i;
            expectedReal += a * c - b * d;
            expectedImag += a * d + b * c;
        }

        v1.setData(data1);
        v2.setData(data2);
        float[] result = v1.dotComplex(v2);

        float errReal = Math.abs(expectedReal - result[0]);
        float errImag = Math.abs(expectedImag - result[1]);
        if (expectedReal == 0.0f) {
            assertTrue("Complex dot product real part error too high", errReal < RELATIVE_TOLERANCE);
        } else {
            assertTrue("Complex dot product real part relative error too high", errReal <= RELATIVE_TOLERANCE * Math.abs(expectedReal));
        }
        if (expectedImag == 0.0f) {
            assertTrue("Complex dot product imaginary part error too high", errImag < RELATIVE_TOLERANCE);
        } else {
            assertTrue("Complex dot product imaginary part relative error too high", errImag <= RELATIVE_TOLERANCE * Math.abs(expectedImag));
        }

        v1.free();
        v2.free();
    }
}