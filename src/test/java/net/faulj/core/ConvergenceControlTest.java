package net.faulj.core;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ConvergenceControlTest {

    @Test
    public void testConvergesOnAbsoluteTolerance() {
        ConvergenceControl control = ConvergenceControl.builder()
            .absoluteTolerance(1e-6)
            .relativeTolerance(1e-4)
            .disableResidualTolerance()
            .stagnationWindow(0)
            .divergenceWindow(0)
            .build();

        ConvergenceControl.State state = control.update(1e-7);

        assertEquals(ConvergenceControl.State.CONVERGED, state);
        assertTrue(control.isConverged());
        assertTrue(control.shouldStop());
    }

    @Test
    public void testStagnationStopsIterations() {
        ConvergenceControl control = ConvergenceControl.builder()
            .absoluteTolerance(1e-12)
            .relativeTolerance(1e-12)
            .disableResidualTolerance()
            .stagnationWindow(3)
            .divergenceWindow(0)
            .maxIterations(100)
            .build();

        for (int i = 0; i < 4; i++) {
            control.update(1.0);
        }

        assertEquals(ConvergenceControl.State.STAGNATED, control.getState());
        assertTrue(control.shouldStop());
        assertFalse(control.isConverged());
    }

    @Test
    public void testDivergenceStopsIterations() {
        ConvergenceControl control = ConvergenceControl.builder()
            .absoluteTolerance(1e-12)
            .relativeTolerance(1e-6)
            .disableResidualTolerance()
            .stagnationWindow(0)
            .divergenceWindow(2)
            .maxIterations(10)
            .build();

        control.update(1.0);
        control.update(2.0);
        ConvergenceControl.State state = control.update(4.0);

        assertEquals(ConvergenceControl.State.DIVERGED, state);
        assertTrue(control.shouldStop());
    }

    @Test
    public void testMaxIterationsStops() {
        ConvergenceControl control = ConvergenceControl.builder()
            .absoluteTolerance(0.0)
            .disableRelativeTolerance()
            .disableResidualTolerance()
            .stagnationWindow(0)
            .divergenceWindow(0)
            .maxIterations(3)
            .build();

        control.update(1.0);
        control.update(1.0);
        ConvergenceControl.State state = control.update(1.0);

        assertEquals(ConvergenceControl.State.MAX_ITERATIONS, state);
        assertTrue(control.shouldStop());
    }

    @Test
    public void testResidualToleranceConvergence() {
        ConvergenceControl control = ConvergenceControl.builder()
            .disableAbsoluteTolerance()
            .disableRelativeTolerance()
            .residualTolerance(1e-6)
            .stagnationWindow(0)
            .divergenceWindow(0)
            .build();

        ConvergenceControl.State state = control.update(10.0, 1e-7);

        assertEquals(ConvergenceControl.State.CONVERGED, state);
        assertTrue(control.isConverged());
    }

    @Test
    public void testResetClearsState() {
        ConvergenceControl control = ConvergenceControl.builder()
            .absoluteTolerance(1e-6)
            .disableResidualTolerance()
            .stagnationWindow(0)
            .divergenceWindow(0)
            .build();

        control.update(1e-7);
        assertTrue(control.isConverged());

        control.reset();

        assertEquals(ConvergenceControl.State.IN_PROGRESS, control.getState());
        assertEquals(0, control.getIterationCount());
        assertTrue(Double.isNaN(control.getLastError()));
    }
}
