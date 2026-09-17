package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AccessKind;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.BufferKind;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/** Focused R1 coverage for executable liveness, coloring, and ownership. */
public class MatrixMemoryPlanningTest {
    @Test
    public void defaultStrategyUsesLinearScanAndExposesMetrics() {
        String previous = System.getProperty(MemoryPlannerStrategy.PROPERTY);
        try {
            System.clearProperty(MemoryPlannerStrategy.PROPERTY);
            CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
                MatrixExpr.input(matrix(2, 2, 1, 2, 3, 4))
                    .add(MatrixExpr.input(matrix(2, 2, 5, 6, 7, 8))));

            PhysicalMemoryPlan plan = compiled.cpuPlan().physicalMemoryPlan();
            assertEquals(1, plan.logicalTemporaryCount());
            assertEquals(1, plan.materializedLogicalTemporaryCount());
            assertEquals(1, plan.physicalSlotCount());
            assertEquals(plan.logicalTemporaryBytesSum(), plan.physicalTemporaryBytes());
            assertEquals(1, plan.allocationCount());
            assertTrue(plan.dump().contains("strategy=linear-scan"));
            assertTrue(compiled.dump().contains("memory planning:"));
        } finally {
            restoreProperty(previous);
        }
    }

    @Test
    public void nonOverlappingCompatibleLifetimesReuseOneSlot() {
        IndependentScaleFixture fixture = independentScales(
            matrix(2, 2, 1, 2, 3, 4), matrix(2, 2, 5, 6, 7, 8));
        PhysicalMemoryPlan plan = fixture.cpuPlan().physicalMemoryPlan();

        assertEquals(2, plan.materializedLogicalTemporaryCount());
        assertEquals(1, plan.physicalSlotCount());
        assertSame(plan.slotFor(fixture.firstTemporary()), plan.slotFor(fixture.secondTemporary()));
        assertEquals(1, plan.logicalToPhysicalReuseCount());
        assertFalse(plan.interferes(fixture.firstTemporary(), fixture.secondTemporary()));
        assertEquals(
            10.0,
            fixture.cpuPlan().execute().get(0, 0),
            0.0);
    }

    @Test
    public void reusedSlotActivatesTheSameMatrixOnlyAfterRelease() {
        IndependentScaleFixture fixture = independentScales(
            matrix(2, 2, 1, 2, 3, 4), matrix(2, 2, 5, 6, 7, 8));
        PhysicalMemoryPlan plan = fixture.cpuPlan().physicalMemoryPlan();
        ExecutionArena arena = new ExecutionArena(plan);

        Matrix first = arena.activate(fixture.firstTemporary());
        arena.release(fixture.firstTemporary());
        Matrix second = arena.activate(fixture.secondTemporary());

        assertSame(first, second);
        arena.closeExcept(null, null);
        arena.closeExcept(null, null);
    }

    @Test
    public void elementwiseStepWritesDirectlyToItsAssignedSlot() {
        IndependentScaleFixture fixture = independentScales(
            matrix(2, 2, 1, 2, 3, 4), matrix(2, 2, 5, 6, 7, 8));
        CpuExecutionPlan plan = fixture.cpuPlan();
        CpuExecutionContext context = new CpuExecutionContext(plan.physicalMemoryPlan());
        for (CpuBufferBinding binding : plan.inputBindings()) {
            context.bind(binding.buffer(), binding.matrix());
        }

        CpuStep step = plan.steps().get(0);
        Matrix assigned = context.allocate(step.outputBuffer());
        ((CpuElementwiseStep) step).execute(context);

        assertSame(assigned, context.value(step.outputBuffer()));
        context.closeOwnedExcept(null, null);
    }

    @Test
    public void overlappingAndSameStepTouchingLifetimesDoNotReuse() {
        OverlapFixture fixture = overlappingScaleAndAdd(
            matrix(2, 2, 1, 2, 3, 4), matrix(2, 2, 5, 6, 7, 8));
        PhysicalMemoryPlan plan = fixture.cpuPlan().physicalMemoryPlan();

        ExecutionLifetime first = plan.lifetime(fixture.firstTemporary());
        ExecutionLifetime second = plan.lifetime(fixture.secondTemporary());
        assertEquals(Integer.valueOf(1), first.lastConsumerStep());
        assertEquals(0, first.startStep());
        assertEquals(1, second.startStep());
        assertTrue(plan.interferes(first, second));
        assertEquals(2, plan.physicalSlotCount());
        assertTrue(plan.dump().contains("slot=P1"));
    }

    @Test
    public void sharedDagProducerLifetimeExtendsThroughBothBranches() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        MatrixExpr shared = MatrixExpr.input(a).add(MatrixExpr.input(b));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            shared.scale(2.0).add(shared.transpose()), OptimizationSemantics.STRICT);

        LogicalBuffer sharedBuffer = compiled.cpuPlan().steps().get(0).outputBuffer();
        ExecutionLifetime lifetime = compiled.cpuPlan().memoryPlan().lifetime(sharedBuffer);
        assertNotNull(lifetime);
        assertEquals(Integer.valueOf(2), lifetime.lastConsumerStep());
        assertEquals(0, lifetime.startStep());
        assertTrue(compiled.cpuPlan().memoryPlan().interferingBuffers(sharedBuffer).size() >= 2);
        assertTrue(compiled.cpuPlan().memoryPlan().peakConcurrentSlots() >= 2);
        assertTrue(compiled.execute().getRowCount() == 2);
    }

    @Test
    public void transposeShapePolicyIsExactButSquareTransposeMayReuse() {
        TransposeFixture rectangular = independentScaleAndTranspose(
            matrix(2, 3, 1, 2, 3, 4, 5, 6), matrix(2, 3, 7, 8, 9, 10, 11, 12));
        assertEquals(2, rectangular.cpuPlan().physicalMemoryPlan().physicalSlotCount());

        TransposeFixture square = independentScaleAndTranspose(
            matrix(2, 2, 1, 2, 3, 4), matrix(2, 2, 5, 6, 7, 8));
        assertEquals(1, square.cpuPlan().physicalMemoryPlan().physicalSlotCount());
    }

    @Test
    public void fusionElisionReceivesNoPhysicalSlot() {
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            MatrixExpr.input(matrix(2, 2, 1, 2, 3, 4)).scale(2.0)
                .add(MatrixExpr.input(matrix(2, 2, 5, 6, 7, 8))));
        LogicalBuffer elided = compiled.cpuPlan().elidedTemporaryBuffers().get(0);

        assertEquals(2, compiled.cpuPlan().logicalTemporaryCount());
        assertEquals(1, compiled.cpuPlan().materializedTemporaryCount());
        assertNull(compiled.cpuPlan().memoryPlan().slotFor(elided));
        assertTrue(compiled.cpuPlan().memoryPlan().dump().contains("elided-by-fusion=1"));
    }

    @Test
    public void outputIsMarkedLiveThroughReturnAndBorrowedInputsHaveNoSlot() {
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            MatrixExpr.input(matrix(2, 2, 1, 2, 3, 4))
                .add(MatrixExpr.input(matrix(2, 2, 5, 6, 7, 8))));
        LogicalBuffer output = compiled.cpuPlan().outputBuffer();
        ExecutionLifetime lifetime = compiled.cpuPlan().memoryPlan().lifetime(output);

        assertTrue(lifetime.liveThroughReturn());
        for (CpuBufferBinding binding : compiled.cpuPlan().inputBindings()) {
            assertNull(compiled.cpuPlan().memoryPlan().slotFor(binding.buffer()));
        }
    }

    @Test
    public void exactShapeAndValueLaneCompatibilityRemainConservative() {
        IndependentScaleFixture differentShapes = independentScales(
            new Matrix(2, 3), new Matrix(3, 2));
        assertEquals(2, differentShapes.cpuPlan().physicalMemoryPlan().physicalSlotCount());

        IndependentScaleFixture complexAndReal = independentScales(
            complexMatrix(2, 2), matrix(2, 2, 5, 6, 7, 8));
        assertEquals(2, complexAndReal.cpuPlan().physicalMemoryPlan().physicalSlotCount());
        assertEquals(
            StorageValueKind.COMPLEX,
            complexAndReal.cpuPlan().physicalMemoryPlan()
                .slotFor(complexAndReal.firstTemporary()).storageClass().valueKind());
        assertEquals(
            StorageValueKind.REAL,
            complexAndReal.cpuPlan().physicalMemoryPlan()
                .slotFor(complexAndReal.secondTemporary()).storageClass().valueKind());
    }

    @Test
    public void heapOffHeapAndUnknownClassesNeverBecomeCompatible() {
        OffHeapMatrix offHeap = new OffHeapMatrix(2, 2);
        try {
            PhysicalStorageClass heap = PhysicalStorageClass.fromMatrix(new Matrix(2, 2));
            PhysicalStorageClass offHeapClass = PhysicalStorageClass.fromMatrix(offHeap);
            PhysicalStorageClass unknown = new PhysicalStorageClass(
                MemorySpace.UNKNOWN, StorageValueKind.UNKNOWN);

            assertFalse(heap.compatibleWith(offHeapClass));
            assertFalse(offHeapClass.compatibleWith(heap));
            assertFalse(unknown.compatibleWith(unknown));
        } finally {
            offHeap.close();
        }
    }

    @Test
    public void offHeapGemmOutputUsesAssignedDestinationAndSurvivesReturn() {
        OffHeapMatrix a = new OffHeapMatrix(2, 2);
        try {
            a.set(0, 0, 1.0);
            a.set(0, 1, 2.0);
            a.set(1, 0, 3.0);
            a.set(1, 1, 4.0);
            Matrix b = matrix(2, 2, 5, 6, 7, 8);
            CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
                MatrixExpr.input(a).matmul(MatrixExpr.input(b)));

            PhysicalBufferSlot slot = compiled.cpuPlan().memoryPlan()
                .slotFor(compiled.cpuPlan().outputBuffer());
            assertEquals(MemorySpace.OFF_HEAP, slot.storageClass().memorySpace());
            Matrix result = compiled.execute();
            assertTrue(result instanceof OffHeapMatrix);
            assertEquals(19.0, result.get(0, 0), 0.0);
            assertTrue(((OffHeapMatrix) result).segment().scope().isAlive());
            ((OffHeapMatrix) result).close();
        } finally {
            if (a.segment().scope().isAlive()) {
                a.close();
            }
        }
    }

    @Test
    public void hiddenOffHeapSlotClosesOnceAtArenaEnd() {
        OffHeapMatrix input = new OffHeapMatrix(2, 2);
        try {
            CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
                MatrixExpr.input(input)
                    .matmul(MatrixExpr.input(matrix(2, 2, 5, 6, 7, 8)))
                    .add(MatrixExpr.input(matrix(2, 2, 1, 1, 1, 1))));
            LogicalBuffer hidden = compiled.cpuPlan().steps().get(0).outputBuffer();
            assertEquals(MemorySpace.OFF_HEAP,
                compiled.cpuPlan().memoryPlan().slotFor(hidden).storageClass().memorySpace());

            ExecutionArena arena = new ExecutionArena(compiled.cpuPlan().physicalMemoryPlan());
            OffHeapMatrix allocated = (OffHeapMatrix) arena.activate(hidden);
            assertTrue(allocated.segment().scope().isAlive());
            arena.release(hidden);
            arena.closeExcept(null, new IllegalStateException("injected execution failure"));
            assertFalse(allocated.segment().scope().isAlive());
            arena.closeExcept(null, null);
        } finally {
            if (input.segment().scope().isAlive()) {
                input.close();
            }
        }
    }

    @Test
    public void legacySelectorRemainsExecutableForABComparison() {
        String previous = System.getProperty(MemoryPlannerStrategy.PROPERTY);
        try {
            System.setProperty(MemoryPlannerStrategy.PROPERTY, "legacy");
            Matrix a = matrix(2, 2, 1, 2, 3, 4);
            Matrix b = matrix(2, 2, 5, 6, 7, 8);
            Matrix actual = MatrixCompiler.compileProgram(
                MatrixExpr.input(a).add(MatrixExpr.input(b))).execute();
            assertEquals(6.0, actual.get(0, 0), 0.0);
        } finally {
            restoreProperty(previous);
        }
    }

    @Test
    public void inputOnlyAndDegeneratePlansNeedNoOwnedSlots() {
        Matrix empty = new Matrix(0, 3);
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(MatrixExpr.input(empty));

        assertEquals(0, compiled.cpuPlan().logicalTemporaryCount());
        assertEquals(0, compiled.cpuPlan().physicalSlotCount());
        assertSame(empty, compiled.execute());
    }

    private static IndependentScaleFixture independentScales(Matrix first, Matrix second) {
        return independentScalesWithShapes(first, second);
    }

    private static IndependentScaleFixture independentScalesWithShapes(Matrix first, Matrix second) {
        MatrixShape firstShape = MatrixShape.from(first);
        MatrixShape secondShape = MatrixShape.from(second);
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer firstInput = external(0, "A", firstShape);
        LogicalBuffer secondInput = external(1, "B", secondShape);
        LogicalBuffer firstTemporary = temporary(2, "tmp0", firstShape);
        LogicalBuffer secondTemporary = temporary(3, "tmp1", secondShape);
        AffineStatement firstStatement = scaleStatement(
            0, firstTemporary, firstInput, firstShape, i, j, 2.0);
        AffineStatement secondStatement = scaleStatement(
            1, secondTemporary, secondInput, secondShape, i, j, 2.0);
        AffineProgram program = program(
            List.of(firstInput, secondInput, firstTemporary, secondTemporary),
            List.of(firstStatement, secondStatement), secondTemporary);
        SchedulePlan schedule = SchedulePlan.initial(program);
        CpuExecutionPlan lowered = CpuLowerer.lower(schedule);
        CpuExecutionPlan withBindings = new CpuExecutionPlan(
            schedule,
            lowered.steps(),
            List.of(new CpuBufferBinding(firstInput, first), new CpuBufferBinding(secondInput, second)),
            List.of(firstTemporary, secondTemporary),
            List.of(firstTemporary, secondTemporary),
            List.of());
        return new IndependentScaleFixture(
            withBindings, firstInput, secondInput, firstTemporary, secondTemporary);
    }

    private static OverlapFixture overlappingScaleAndAdd(Matrix first, Matrix second) {
        MatrixShape shape = MatrixShape.from(first);
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer firstInput = external(0, "A", shape);
        LogicalBuffer secondInput = external(1, "B", shape);
        LogicalBuffer firstTemporary = temporary(2, "tmp0", shape);
        LogicalBuffer secondTemporary = temporary(3, "tmp1", shape);
        AffineStatement scale = scaleStatement(0, firstTemporary, firstInput, shape, i, j, 2.0);
        AffineExpr[] indices = indices(i, j);
        AffineStatement add = new AffineStatement(
            1,
            StatementKind.ADD,
            domain(shape, i, j),
            List.of(
                new AffineAccess(secondTemporary, AccessKind.WRITE, List.of(indices)),
                new AffineAccess(firstTemporary, AccessKind.READ, List.of(indices)),
                new AffineAccess(secondInput, AccessKind.READ, List.of(indices))),
            "%3[i,j] = %2[i,j] + %1[i,j]");
        AffineProgram program = program(
            List.of(firstInput, secondInput, firstTemporary, secondTemporary),
            List.of(scale, add), secondTemporary);
        SchedulePlan schedule = SchedulePlan.initial(program);
        CpuExecutionPlan lowered = CpuLowerer.lower(schedule);
        CpuExecutionPlan withBindings = new CpuExecutionPlan(
            schedule,
            lowered.steps(),
            List.of(new CpuBufferBinding(firstInput, first), new CpuBufferBinding(secondInput, second)),
            List.of(firstTemporary, secondTemporary),
            List.of(firstTemporary, secondTemporary),
            List.of());
        return new OverlapFixture(withBindings, firstTemporary, secondTemporary);
    }

    private static TransposeFixture independentScaleAndTranspose(Matrix scaleInput,
                                                                  Matrix transposeInput) {
        MatrixShape scaleShape = MatrixShape.from(scaleInput);
        MatrixShape transposeShape = MatrixShape.from(transposeInput);
        MatrixShape transposeOutputShape = new MatrixShape(
            transposeShape.columns(), transposeShape.rows());
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer scaleBuffer = external(0, "A", scaleShape);
        LogicalBuffer transposeBuffer = external(1, "B", transposeShape);
        LogicalBuffer scaleTemporary = temporary(2, "tmp0", scaleShape);
        LogicalBuffer transposeTemporary = temporary(3, "tmp1", transposeOutputShape);
        AffineStatement scale = scaleStatement(
            0, scaleTemporary, scaleBuffer, scaleShape, i, j, 2.0);
        AffineStatement transpose = new AffineStatement(
            1,
            StatementKind.TRANSPOSE,
            domain(transposeShape, i, j),
            List.of(
                new AffineAccess(transposeTemporary, AccessKind.WRITE,
                    List.of(AffineExpr.variable(j), AffineExpr.variable(i))),
                new AffineAccess(transposeBuffer, AccessKind.READ, List.of(indices(i, j)))),
            "%3[j,i] = %1[i,j]");
        AffineProgram program = program(
            List.of(scaleBuffer, transposeBuffer, scaleTemporary, transposeTemporary),
            List.of(scale, transpose), transposeTemporary);
        SchedulePlan schedule = SchedulePlan.initial(program);
        CpuExecutionPlan lowered = CpuLowerer.lower(schedule);
        CpuExecutionPlan withBindings = new CpuExecutionPlan(
            schedule,
            lowered.steps(),
            List.of(new CpuBufferBinding(scaleBuffer, scaleInput),
                new CpuBufferBinding(transposeBuffer, transposeInput)),
            List.of(scaleTemporary, transposeTemporary),
            List.of(scaleTemporary, transposeTemporary),
            List.of());
        return new TransposeFixture(withBindings);
    }

    private static AffineStatement scaleStatement(int id,
                                                   LogicalBuffer output,
                                                   LogicalBuffer input,
                                                   MatrixShape shape,
                                                   AffineVariable i,
                                                   AffineVariable j,
                                                   double factor) {
        AffineExpr[] indices = indices(i, j);
        return new AffineStatement(
            id,
            StatementKind.SCALE,
            domain(shape, i, j),
            List.of(
                new AffineAccess(output, AccessKind.WRITE, List.of(indices)),
                new AffineAccess(input, AccessKind.READ, List.of(indices))),
            "%" + output.id() + "[i,j] = " + factor + " * %" + input.id() + "[i,j]");
    }

    private static AffineProgram program(List<LogicalBuffer> buffers,
                                         List<AffineStatement> statements,
                                         LogicalBuffer output) {
        return new AffineProgram(
            OptimizationSemantics.STRICT,
            buffers,
            List.of(new AffineVariable("i"), new AffineVariable("j")),
            statements,
            output,
            new net.faulj.compiler.matrix.affine.DependenceGraph(statements));
    }

    private static net.faulj.compiler.matrix.affine.IterationDomain domain(
        MatrixShape shape, AffineVariable i, AffineVariable j) {
        return net.faulj.compiler.matrix.affine.IterationDomain.of(
            net.faulj.compiler.matrix.affine.IterationDomain.range(i, 0, shape.rows()),
            net.faulj.compiler.matrix.affine.IterationDomain.range(j, 0, shape.columns()));
    }

    private static AffineExpr[] indices(AffineVariable i, AffineVariable j) {
        return new AffineExpr[]{AffineExpr.variable(i), AffineExpr.variable(j)};
    }

    private static LogicalBuffer external(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(
            id, name, BufferKind.EXTERNAL_INPUT, BufferOwnership.BORROWED,
            MemorySpace.HEAP, shape);
    }

    private static LogicalBuffer temporary(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(
            id, name, BufferKind.TEMPORARY, BufferOwnership.OWNED,
            MemorySpace.HEAP, shape);
    }

    private static Matrix matrix(int rows, int columns, double... values) {
        Matrix result = new Matrix(rows, columns);
        System.arraycopy(values, 0, result.getRawData(), 0, values.length);
        return result;
    }

    private static Matrix complexMatrix(int rows, int columns) {
        Matrix result = matrix(rows, columns, 1, 2, 3, 4);
        result.ensureImagData()[0] = 0.5;
        return result;
    }

    private static void restoreProperty(String previous) {
        if (previous == null) {
            System.clearProperty(MemoryPlannerStrategy.PROPERTY);
        } else {
            System.setProperty(MemoryPlannerStrategy.PROPERTY, previous);
        }
    }

    private record IndependentScaleFixture(CpuExecutionPlan cpuPlan,
                                           LogicalBuffer firstInput,
                                           LogicalBuffer secondInput,
                                           LogicalBuffer firstTemporary,
                                           LogicalBuffer secondTemporary) {
    }

    private record OverlapFixture(CpuExecutionPlan cpuPlan,
                                  LogicalBuffer firstTemporary,
                                  LogicalBuffer secondTemporary) {
    }

    private record TransposeFixture(CpuExecutionPlan cpuPlan) {
    }
}
