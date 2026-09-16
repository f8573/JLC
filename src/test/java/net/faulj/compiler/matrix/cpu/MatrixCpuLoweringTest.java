package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;
import java.util.Map;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;
import net.faulj.kernels.gemm.Gemm;

/** Focused M4 coverage for executable CPU lowering and the complete pipeline. */
public class MatrixCpuLoweringTest {
    @Test
    public void completeCompilePipelineIsInspectable() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            MatrixExpr.input("A", a).matmul(MatrixExpr.input("B", b)),
            OptimizationSemantics.RELAXED);

        assertNotNull(program.expressionPlan());
        assertNotNull(program.affineProgram());
        assertNotNull(program.dependenceGraph());
        assertNotNull(program.schedule());
        assertNotNull(program.cpuPlan());
        assertTrue(program.dump().contains("expression plan:"));
        assertTrue(program.dump().contains("cpu execution plan:"));
    }

    @Test
    public void cpuPlanDumpIsDeterministic() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        MatrixExpr expression = MatrixExpr.input("A", a).add(MatrixExpr.input("B", b));

        assertEquals(
            MatrixCompiler.compileProgram(expression).cpuPlan().dump(),
            MatrixCompiler.compileProgram(expression).cpuPlan().dump());
    }

    @Test
    public void symbolicInputExecutionIsRejected() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 2))
            .add(MatrixExpr.symbolicInput("B", new MatrixShape(2, 2)));
        try {
            MatrixCompiler.compileProgram(expression).execute();
            fail("symbolic CPU plan should not execute");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("symbolic"));
        }
    }

    @Test
    public void addExecutesThroughAnExplicitElementwiseStep() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(2, 3, 10, 20, 30, 40, 50, 60);
        MatrixExpr expression = MatrixExpr.input(a).add(MatrixExpr.input(b));

        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(expression);
        assertEquals(CpuStepKind.ELEMENTWISE, compiled.cpuPlan().steps().get(0).kind());
        assertMatrixEquals(a.add(b), compiled.execute());
    }

    @Test
    public void scaleExecutesThroughAnExplicitElementwiseStep() {
        Matrix a = matrix(2, 2, 1, -2, 3, -4);
        MatrixExpr expression = MatrixExpr.input(a).scale(2.5);

        assertMatrixEquals(a.multiplyScalar(2.5), MatrixCompiler.compileProgram(expression).execute());
    }

    @Test
    public void transposeExecutesThroughAnExplicitLoopStep() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        MatrixExpr expression = MatrixExpr.input(a).transpose();

        assertEquals(CpuStepKind.TRANSPOSE, MatrixCompiler.compileProgram(expression)
            .cpuPlan().steps().get(0).kind());
        assertMatrixEquals(a.transpose(), MatrixCompiler.compileProgram(expression).execute());
    }

    @Test
    public void matMulUsesOneOpaqueProductionGemmStep() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(3, 2, 7, 8, 9, 10, 11, 12);
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            MatrixExpr.input(a).matmul(MatrixExpr.input(b)));

        assertEquals(1L, compiled.cpuPlan().gemmStepCount());
        assertEquals(CpuStepKind.GEMM, compiled.cpuPlan().steps().get(0).kind());
        assertMatrixEquals(Gemm.multiply(a, b), compiled.execute());
    }

    @Test
    public void strictChainExecutesItsOriginalAssociation() {
        Matrix a = matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(2, 4, 1, 2, 3, 4, 5, 6, 7, 8);
        Matrix c = matrix(4, 2, 2, 1, 4, 3, 6, 5, 8, 7);
        MatrixExpr expression = MatrixExpr.input(a).matmul(MatrixExpr.input(b))
            .matmul(MatrixExpr.input(c));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);

        assertEquals(2L, compiled.cpuPlan().gemmStepCount());
        assertMatrixEquals(MatrixCompiler.evaluate(expression, OptimizationSemantics.STRICT), compiled.execute());
    }

    @Test
    public void relaxedChainExecutesTheSelectedReassociation() {
        Matrix a = matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(2, 4, 1, 2, 3, 4, 5, 6, 7, 8);
        Matrix c = matrix(4, 2, 2, 1, 4, 3, 6, 5, 8, 7);
        MatrixExpr expression = MatrixExpr.input(a).matmul(MatrixExpr.input(b))
            .matmul(MatrixExpr.input(c));
        CompiledMatrixProgram strict = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);
        CompiledMatrixProgram relaxed = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.RELAXED);

        assertNotEquals(strict.expressionPlan().expression(), relaxed.expressionPlan().expression());
        assertMatrixEquals(strict.execute(), relaxed.execute());
        assertEquals(MatrixCompiler.evaluate(expression, OptimizationSemantics.RELAXED).get(0, 0),
            relaxed.execute().get(0, 0), 1e-10);
    }

    @Test
    public void fusedScaleAddUsesOneOutputLoopAndElidesScaleTemporary() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(2, 3, 10, 20, 30, 40, 50, 60);
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(expression);

        assertTrue(compiled.cpuPlan().hasFusedElementwiseStep());
        assertEquals(1, compiled.cpuPlan().elidedTemporaryCount());
        assertEquals(1, compiled.cpuPlan().materializedTemporaryCount());
        assertMatrixEquals(a.multiplyScalar(2.0).add(b), compiled.execute());
    }

    @Test
    public void sharedDagComputesProducerOnce() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        MatrixExpr x = MatrixExpr.input(a).matmul(MatrixExpr.input(b));
        MatrixExpr expression = x.add(x);
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(expression);

        assertEquals(1L, compiled.cpuPlan().gemmStepCount());
        assertEquals(2, compiled.cpuPlan().temporaryCount());
        Matrix expected = Gemm.multiply(a, b);
        assertMatrixEquals(expected.add(expected), compiled.execute());
    }

    @Test
    public void inputsRemainUnmodifiedAndTemporariesAreExecutionOwned() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        Matrix aBefore = a.copy();
        Matrix bBefore = b.copy();
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            MatrixExpr.input(a).matmul(MatrixExpr.input(b)));

        assertEquals(BufferOwnership.BORROWED, compiled.cpuPlan().inputBindings().get(0)
            .buffer().ownership());
        for (LogicalBuffer buffer : compiled.cpuPlan().materializedTemporaryBuffers()) {
            assertEquals(BufferOwnership.OWNED, buffer.ownership());
        }
        Matrix result = compiled.execute();
        assertTrue(result != a && result != b);
        assertMatrixEquals(aBefore, a);
        assertMatrixEquals(bBefore, b);
    }

    @Test
    public void offHeapInputUsesExistingPublicMatrixAccess() {
        OffHeapMatrix a = new OffHeapMatrix(2, 2);
        try {
            a.set(0, 0, 1);
            a.set(0, 1, 2);
            a.set(1, 0, 3);
            a.set(1, 1, 4);
            Matrix b = matrix(2, 2, 5, 6, 7, 8);

            Matrix actual = MatrixCompiler.compileProgram(
                MatrixExpr.input(a).add(MatrixExpr.input(b))).execute();
            assertMatrixEquals(a.add(b), actual);
        } finally {
            a.close();
        }
    }

    @Test
    public void transformedScheduleInterchangeIsReflectedInExecution() {
        Matrix a = matrix(3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        Matrix b = matrix(3, 4, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        ExecutionPlan plan = MatrixCompiler.compile(MatrixExpr.input(a).add(MatrixExpr.input(b)));
        AffineProgram affine = AffineProgram.lower(plan);
        SchedulePlan schedule = SchedulePlan.initial(affine).interchange(0, "i", "j").schedule();
        CpuExecutionPlan cpu = CpuLowerer.lower(schedule, plan);

        assertEquals(List.of("j", "i"), schedule.region(0).band().loops().stream()
            .map(loop -> loop.inductionVariable().name()).toList());
        assertMatrixEquals(a.add(b), cpu.execute());
    }

    @Test
    public void stripMinedRemainderExecutesCorrectly() {
        Matrix a = matrix(5, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        Matrix b = matrix(5, 3, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        ExecutionPlan plan = MatrixCompiler.compile(MatrixExpr.input(a).add(MatrixExpr.input(b)));
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(plan))
            .stripMine(0, "i", 2).schedule();

        assertMatrixEquals(a.add(b), CpuLowerer.lower(schedule, plan).execute());
    }

    @Test
    public void explicitFusedScheduleExecutesCorrectly() {
        Matrix a = matrix(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        Matrix b = matrix(3, 3, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        MatrixExpr expression = MatrixExpr.input(a).scale(3.0).add(MatrixExpr.input(b));
        ExecutionPlan plan = MatrixCompiler.compile(expression);
        AffineProgram affine = AffineProgram.lower(plan);
        SchedulePlan schedule = SchedulePlan.initial(affine).fusion(0, 1).schedule();
        CpuExecutionPlan cpu = CpuLowerer.lower(schedule, plan);

        assertTrue(cpu.hasFusedElementwiseStep());
        assertMatrixEquals(a.multiplyScalar(3.0).add(b), cpu.execute());
    }

    @Test
    public void symmetricScaleAddFusionPreservesAddOperandOrder() {
        Matrix a = matrix(2, 2, 0.1, 0.2, 0.3, 0.4);
        Matrix b = matrix(2, 2, 0.5, 0.6, 0.7, 0.8);
        MatrixExpr expression = MatrixExpr.input(a)
            .add(MatrixExpr.input(b).scale(2.0));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(expression);

        assertTrue(compiled.cpuPlan().hasFusedElementwiseStep());
        CpuFusedElementwiseStep fused = (CpuFusedElementwiseStep) compiled.cpuPlan().steps().get(0);
        assertFalse(fused.scaledOperandFirst());
        assertMatrixEquals(a.add(b.multiplyScalar(2.0)), compiled.execute());
    }

    @Test
    public void planOnlyLoweringAcceptsExplicitExternalBindings() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        ExecutionPlan plan = MatrixCompiler.compile(MatrixExpr.input(a).add(MatrixExpr.input(b)));
        AffineProgram affine = AffineProgram.lower(plan);
        CpuExecutionPlan cpu = CpuLowerer.lower(SchedulePlan.initial(affine));
        Map<LogicalBuffer, Matrix> bindings = Map.of(
            cpu.inputBindings().get(0).buffer(), a,
            cpu.inputBindings().get(1).buffer(), b);

        assertMatrixEquals(a.add(b), cpu.execute(bindings));
    }

    @Test
    public void parallelAnnotationFallsBackToSerialExecution() {
        Matrix a = matrix(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        Matrix b = matrix(3, 3, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        ExecutionPlan plan = MatrixCompiler.compile(MatrixExpr.input(a).add(MatrixExpr.input(b)));
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(plan))
            .parallel(0, "i").schedule();
        CpuExecutionPlan cpu = CpuLowerer.lower(schedule, plan);

        assertTrue(cpu.hasParallelAnnotations());
        assertFalse(cpu.realizesParallelAnnotations());
        assertMatrixEquals(a.add(b), cpu.execute());
    }

    @Test
    public void vectorAnnotationFallsBackToScalarExecution() {
        Matrix a = matrix(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        Matrix b = matrix(3, 3, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        ExecutionPlan plan = MatrixCompiler.compile(MatrixExpr.input(a).add(MatrixExpr.input(b)));
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(plan))
            .vector(0, "j").schedule();
        CpuExecutionPlan cpu = CpuLowerer.lower(schedule, plan);

        assertTrue(cpu.hasVectorAnnotations());
        assertFalse(cpu.realizesVectorAnnotations());
        assertMatrixEquals(a.add(b), cpu.execute());
    }

    @Test
    public void strictReductionLegalityRemainsIntact() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 2)));
        ExecutionPlan plan = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(plan));

        assertEquals(LegalityStatus.ILLEGAL, schedule.interchange(1, "j", "k").status());
        assertEquals(LegalityStatus.ILLEGAL, schedule.parallel(1, "k").status());
        assertEquals(LegalityStatus.ILLEGAL, schedule.vector(1, "k").status());
    }

    @Test
    public void relaxedReductionMetadataRemainsExplicit() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 2)));
        ExecutionPlan plan = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(plan))
            .interchange(1, "j", "k").schedule();

        assertTrue(schedule.hasReductionReassociationMetadata());
        assertTrue(CpuLowerer.lower(schedule).dump().contains("output:"));
    }

    @Test
    public void fastSemanticsRemainExplicitAndExecutableLikeRelaxed() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        Matrix c = matrix(2, 2, 2, 1, 4, 3);
        MatrixExpr expression = MatrixExpr.input(a).matmul(MatrixExpr.input(b))
            .matmul(MatrixExpr.input(c));
        CompiledMatrixProgram fast = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.FAST);

        assertEquals(OptimizationSemantics.FAST, fast.cpuPlan().semantics());
        assertMatrixEquals(MatrixCompiler.evaluate(expression, OptimizationSemantics.FAST), fast.execute());
    }

    private static Matrix matrix(int rows, int columns, double... values) {
        assertEquals(rows * columns, values.length);
        Matrix result = new Matrix(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                result.set(row, column, values[row * columns + column]);
            }
        }
        return result;
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int row = 0; row < expected.getRowCount(); row++) {
            for (int column = 0; column < expected.getColumnCount(); column++) {
                assertEquals(
                    "at [" + row + "," + column + "]",
                    expected.get(row, column), actual.get(row, column), 1e-9);
            }
        }
    }
}
