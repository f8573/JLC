package net.faulj.core;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests for diagnostic metric analysis on matrices.
 */
public class DiagnosticMetricsTest {

    /**
     * Validate diagnostics for a symmetric positive definite matrix.
     */
    @Test
    public void testDiagnosticsForSPDMatrix() {
        Matrix A = new Matrix(new double[][]{
            {4, 1},
            {1, 3}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);

        assertTrue(diag.isSquare());
        assertTrue(diag.isSymmetric());
        assertNotNull(diag.getSvd());
        assertNotEquals(DiagnosticMetrics.Status.ERROR, diag.getSvd().getStatus());
        assertNotNull(diag.getCholesky());
        assertNotEquals(DiagnosticMetrics.Status.ERROR, diag.getCholesky().getStatus());
        assertNotNull(diag.getInverse());
        assertEquals(DiagnosticMetrics.Status.OK, diag.getInverse().getStatus());
        assertEquals(Integer.valueOf(2), diag.getRank());
        assertEquals(Integer.valueOf(0), diag.getNullity());
    }

    /**
     * Validate diagnostics for a singular matrix.
     */
    @Test
    public void testDiagnosticsForSingularMatrix() {
        Matrix A = new Matrix(new double[][]{
            {1, 2},
            {2, 4}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);

        assertEquals(DiagnosticMetrics.Status.ERROR, diag.getInverse().getStatus());
        assertEquals(DiagnosticMetrics.Status.WARNING, diag.getLu().getStatus());
        assertEquals(Integer.valueOf(1), diag.getRank());
        assertEquals(Integer.valueOf(1), diag.getNullity());
    }

    /**
     * Validate diagnostics for a rectangular matrix.
     */
    @Test
    public void testDiagnosticsForRectangularMatrix() {
        Matrix A = new Matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        DiagnosticMetrics.MatrixDiagnostics diag = DiagnosticMetrics.analyze(A);

        assertFalse(diag.isSquare());
        assertEquals(DiagnosticMetrics.Status.ERROR, diag.getLu().getStatus());
        assertEquals(DiagnosticMetrics.Status.ERROR, diag.getInverse().getStatus());
        assertNotEquals(DiagnosticMetrics.Status.ERROR, diag.getSvd().getStatus());
        assertEquals(Integer.valueOf(2), diag.getRank());
        assertEquals(Integer.valueOf(1), diag.getNullity());
    }
}
