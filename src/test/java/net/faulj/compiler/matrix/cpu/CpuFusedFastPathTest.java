package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.*;

import java.util.List;
import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

public class CpuFusedFastPathTest {
    private static CpuFusedElementwiseStep fused(CompiledMatrixProgram program) {
        return (CpuFusedElementwiseStep) program.cpuPlan().steps().get(0);
    }

    @Test
    public void realRectangularIdentityUsesFastPathAndMatchesEagerFactors() {
        double[] factors = {0.0, 1.0, -1.0, 2.5, Double.NaN,
            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        Matrix a = Matrix.wrap(new double[] {1.0, -2.0, -0.0, 3.5, -7.0, 0.0}, 2, 3);
        Matrix b = Matrix.wrap(new double[] {-3.0, 4.0, 0.0, -1.5, 8.0, -0.0}, 2, 3);
        double[] beforeA = a.getRawData().clone();
        double[] beforeB = b.getRawData().clone();
        CpuFusedElementwiseStep original = fused(MatrixCompiler.compileProgram(
            MatrixExpr.input(a).scale(2.5).add(MatrixExpr.input(b))));
        for (double factor : factors) {
            // Zero and one may be simplified by the expression optimizer before M4.
            CpuFusedElementwiseStep step = new CpuFusedElementwiseStep(
                original.id(), original.scaleStatementId(), original.addStatementId(),
                original.scheduleBand(), original.outputBuffer(), original.scaledOperand(),
                original.addOperand(), original.eliminatedBuffer(), factor,
                original.scaledOperandFirst(), original.outputAccess(), original.scaleAccess(),
                original.addAccess());
            CpuExecutionContext context = new CpuExecutionContext();
            context.bind(original.scaledOperand(), a);
            context.bind(original.addOperand(), b);
            step.execute(context);
            Matrix actual = context.value(original.outputBuffer());
            assertEquals(CpuFusedElementwiseStep.ExecutionPath.FAST_REAL_IDENTITY,
                step.lastExecutionPath());
            assertNotSame(a, actual);
            assertNotSame(b, actual);
            Matrix expected = a.multiplyScalar(factor).add(b);
            for (int p = 0; p < actual.getRawData().length; p++) {
                assertEquals("element " + p + " factor " + factor,
                    Double.doubleToRawLongBits(expected.getRawData()[p]),
                    Double.doubleToRawLongBits(actual.getRawData()[p]));
            }
        }
        assertArrayEquals(beforeA, a.getRawData(), 0.0);
        assertArrayEquals(beforeB, b.getRawData(), 0.0);
    }

    @Test
    public void complexAndOffHeapInputsUseGenericFallback() {
        Matrix complex = Matrix.wrap(new double[] {-0.0, 2.0},
            new double[] {-0.0, Double.NaN}, 1, 2);
        Matrix real = Matrix.wrap(new double[] {0.0, Double.POSITIVE_INFINITY}, 1, 2);
        CompiledMatrixProgram complexProgram = MatrixCompiler.compileProgram(
            MatrixExpr.input(complex).scale(Double.NEGATIVE_INFINITY)
                .add(MatrixExpr.input(real)));
        Matrix actual = complexProgram.execute();
        assertEquals(CpuFusedElementwiseStep.ExecutionPath.GENERIC_SCHEDULE,
            fused(complexProgram).lastExecutionPath());
        Matrix expected = complex.multiplyScalar(Double.NEGATIVE_INFINITY).add(real);
        for (int p = 0; p < 2; p++) {
            assertEquals(Double.doubleToRawLongBits(expected.getRawData()[p]),
                Double.doubleToRawLongBits(actual.getRawData()[p]));
            assertEquals(Double.doubleToRawLongBits(expected.getRawImagData()[p]),
                Double.doubleToRawLongBits(actual.getRawImagData()[p]));
        }

        OffHeapMatrix offHeap = new OffHeapMatrix(1, 2);
        try {
            offHeap.set(0, 0, 2.0);
            offHeap.set(0, 1, -3.0);
            CompiledMatrixProgram offHeapProgram = MatrixCompiler.compileProgram(
                MatrixExpr.input(offHeap).scale(2.5).add(MatrixExpr.input(real)));
            Matrix offHeapActual = offHeapProgram.execute();
            assertEquals(CpuFusedElementwiseStep.ExecutionPath.GENERIC_SCHEDULE,
                fused(offHeapProgram).lastExecutionPath());
            Matrix offHeapExpected = offHeap.multiplyScalar(2.5).add(real);
            for (int p = 0; p < 2; p++) {
                assertEquals(offHeapExpected.getRawData()[p], offHeapActual.getRawData()[p], 0.0);
            }
        } finally {
            offHeap.close();
        }
    }

    @Test
    public void transformedScheduleUsesGenericFallback() {
        Matrix a = Matrix.wrap(new double[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Matrix b = Matrix.wrap(new double[] {6, 5, 4, 3, 2, 1}, 2, 3);
        CpuFusedElementwiseStep original = fused(MatrixCompiler.compileProgram(
            MatrixExpr.input(a).scale(2.5).add(MatrixExpr.input(b))));
        ScheduleBand band = original.scheduleBand();
        ScheduleLoop row = band.loop(0);
        AffineVariable shifted = AffineVariable.named("shiftedRow");
        ScheduleLoop shiftedRow = row.copyAs(shifted, row.semanticVariable(),
            1, 3, 1, List.of(), AffineExpr.variable(shifted).add(-1), null,
            "shifted schedule");
        ScheduleBand transformed = band.withLoops(List.of(shiftedRow, band.loop(1)));
        CpuFusedElementwiseStep step = new CpuFusedElementwiseStep(
            original.id(), original.scaleStatementId(), original.addStatementId(), transformed,
            original.outputBuffer(), original.scaledOperand(), original.addOperand(),
            original.eliminatedBuffer(), original.factor(), original.scaledOperandFirst(),
            original.outputAccess(), original.scaleAccess(), original.addAccess());
        CpuExecutionContext context = new CpuExecutionContext();
        context.bind(original.scaledOperand(), a);
        context.bind(original.addOperand(), b);
        step.execute(context);
        assertEquals(CpuFusedElementwiseStep.ExecutionPath.GENERIC_SCHEDULE,
            step.lastExecutionPath());
        assertArrayEquals(a.multiplyScalar(2.5).add(b).getRawData(),
            context.value(original.outputBuffer()).getRawData(), 0.0);
    }

    @Test
    public void zeroSizedIdentityUsesFastPath() {
        Matrix a = new Matrix(0, 3);
        Matrix b = new Matrix(0, 3);
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            MatrixExpr.input(a).scale(2.5).add(MatrixExpr.input(b)));
        Matrix actual = program.execute();
        assertEquals(CpuFusedElementwiseStep.ExecutionPath.FAST_REAL_IDENTITY,
            fused(program).lastExecutionPath());
        assertEquals(0, actual.getRawData().length);
    }

    @Test
    public void sharedFusedProducerRemainsCorrect() {
        Matrix a = Matrix.wrap(new double[] {1, -2, 3, -4}, 2, 2);
        Matrix b = Matrix.wrap(new double[] {4, 3, -2, -1}, 2, 2);
        MatrixExpr producer = MatrixExpr.input(a).scale(2.5).add(MatrixExpr.input(b));
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(producer.add(producer));
        Matrix expectedProducer = a.multiplyScalar(2.5).add(b);
        assertArrayEquals(expectedProducer.add(expectedProducer).getRawData(),
            program.execute().getRawData(), 0.0);
    }
}
