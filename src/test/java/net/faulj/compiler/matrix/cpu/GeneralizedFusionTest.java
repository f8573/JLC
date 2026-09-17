package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.BufferKind;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.DependenceGraph;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;

/** Focused R2 coverage for legality-aware generalized fusion. */
public class GeneralizedFusionTest {
    @Test
    public void fusesScaleTransposeScaleAddIntoOneRegion() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(3, 2, 10, 20, 30, 40, 50, 60);
        double[] aBefore = a.getRawData().clone();
        double[] bBefore = b.getRawData().clone();
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0).transpose()
            .scale(-0.5).add(MatrixExpr.input(b));

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        CpuExecutionPlan plan = program.cpuPlan();

        assertEquals(1, plan.stepCount());
        assertTrue(plan.steps().get(0) instanceof CpuFusedRegionStep);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) plan.steps().get(0);
        assertEquals(4, fused.regionPlan().statements().size());
        assertEquals(3, fused.eliminatedBuffers().size());
        assertEquals(2, fused.inputBuffers().size());
        assertEquals(CpuFusedRegionStep.ExecutionPath.FAST_REAL_DENSE,
            executeAndPath(program, fused));
        assertMatrixEquals(a.multiplyScalar(2.0).transpose().multiplyScalar(-0.5).add(b),
            program.execute());
        assertArrayEquals(aBefore, a.getRawData(), 0.0);
        assertArrayEquals(bBefore, b.getRawData(), 0.0);
        assertTrue(plan.dump().contains("legality=LEGAL"));
        assertTrue(plan.dump().contains("profitability="));
    }

    @Test
    public void fusesSharedScalarDagWithoutMaterializingInternalValues() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        MatrixExpr shared = MatrixExpr.input(a).scale(3.0);
        CompiledMatrixProgram program = compile(shared.add(shared), FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) program.cpuPlan().steps().get(0);

        assertEquals(2, fused.regionPlan().statements().size());
        assertEquals(1, fused.regionPlan().scalarProgram().sharedProducerCount());
        assertEquals(1, program.cpuPlan().elidedTemporaryCount());
        assertMatrixEquals(a.multiplyScalar(3.0).add(a.multiplyScalar(3.0)), program.execute());
    }

    @Test
    public void preservesAddThenScaleAssociationAndSupportsLongChains() {
        Matrix a = matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(3, 2, 7, 8, 9, 10, 11, 12);
        Matrix c = matrix(3, 2, 2, 4, 6, 8, 10, 12);
        MatrixExpr expression = MatrixExpr.input(a).add(MatrixExpr.input(b))
            .scale(0.25).add(MatrixExpr.input(c)).scale(-3.0)
            .scale(0.5).add(MatrixExpr.input(b)).scale(2.0);

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) program.cpuPlan().steps().get(0);
        assertEquals(7, fused.regionPlan().statements().size());
        assertEquals(
            List.of(
                ScalarFusionProgram.Opcode.LOAD,
                ScalarFusionProgram.Opcode.LOAD,
                ScalarFusionProgram.Opcode.ADD,
                ScalarFusionProgram.Opcode.SCALE),
            fused.scalarProgram().nodes().subList(0, 4).stream()
                .map(ScalarFusionProgram.ScalarNode::opcode).toList());
        Matrix expected = a.add(b).multiplyScalar(0.25).add(c).multiplyScalar(-3.0)
            .multiplyScalar(0.5).add(b).multiplyScalar(2.0);
        assertMatrixEquals(expected, program.execute());
    }

    @Test
    public void rectangularTransposePipelineUsesComposedMaps() {
        Matrix a = matrix(37, 11, sequence(37 * 11));
        Matrix b = matrix(11, 37, sequence(11 * 37, 1000));
        MatrixExpr expression = MatrixExpr.input(a).transpose().scale(1.5)
            .add(MatrixExpr.input(b)).transpose().scale(-2.0);

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) program.cpuPlan().steps().get(0);
        assertEquals(5, fused.regionPlan().statements().size());
        assertEquals(2, fused.regionPlan().scalarProgram().nodes().stream()
            .filter(node -> node.opcode() == ScalarFusionProgram.Opcode.TRANSPOSE).count());
        assertMatrixEquals(
            a.transpose().multiplyScalar(1.5).add(b).transpose().multiplyScalar(-2.0),
            program.execute());
    }

    @Test
    public void twoTransposeMapsComposeThroughAnElementwiseOperation() {
        Matrix a = matrix(2, 3, 1, 2, 3, 4, 5, 6);
        Matrix b = matrix(2, 3, 10, 20, 30, 40, 50, 60);
        MatrixExpr expression = MatrixExpr.input(a).transpose().scale(2.0).transpose()
            .add(MatrixExpr.input(b));

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) program.cpuPlan().steps().get(0);

        assertEquals(4, fused.regionPlan().statements().size());
        assertEquals(2, fused.scalarProgram().nodes().stream()
            .filter(node -> node.opcode() == ScalarFusionProgram.Opcode.TRANSPOSE).count());
        assertEquals(3, fused.eliminatedBuffers().size());
        assertMatrixEquals(a.transpose().multiplyScalar(2.0).transpose().add(b), program.execute());
    }

    @Test
    public void tileScheduleFallsBackToGenericWithoutAllocatingPerPointObjects() {
        Matrix a = matrix(5, 3, sequence(15));
        Matrix b = matrix(5, 3, sequence(15, 50));
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b));
        var expressionPlan = MatrixCompiler.compile(expression);
        var affine = net.faulj.compiler.matrix.affine.AffineProgram.lower(expressionPlan);
        var schedule = net.faulj.compiler.matrix.schedule.SchedulePlan.initial(affine)
            .stripMine(1, "i", 2).schedule();
        CpuExecutionPlan plan = CpuLowerer.lower(schedule, expressionPlan, FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) plan.steps().get(0);

        assertMatrixEquals(a.multiplyScalar(2.0).add(b), plan.execute());
        assertEquals(CpuFusedRegionStep.ExecutionPath.GENERIC_SCHEDULE,
            fused.lastExecutionPath());
    }

    @Test
    public void complexInputUsesGenericCompiledScheduleAndPreservesLanes() {
        Matrix a = new Matrix(
            new double[][]{{1, -2}, {3, -4}},
            new double[][]{{0.5, -0.0}, {Double.NaN, 2.0}});
        Matrix b = matrix(2, 2, 10, 20, 30, 40);
        CompiledMatrixProgram program = compile(
            MatrixExpr.input(a).scale(-2.0).add(MatrixExpr.input(b)),
            FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) program.cpuPlan().steps().get(0);
        Matrix expected = a.multiplyScalar(-2.0).add(b);
        Matrix actual = program.execute();

        assertEquals(CpuFusedRegionStep.ExecutionPath.GENERIC_SCHEDULE,
            fused.lastExecutionPath());
        assertTrue(actual.hasImagData());
        assertMatrixEquals(expected, actual);
    }

    @Test
    public void complexLanePresenceVariantsMatchEagerOperations() {
        Matrix real = matrix(2, 2, 1, -2, 3, -4);
        Matrix complex = new Matrix(
            new double[][]{{5, 6}, {7, 8}},
            new double[][]{{0.0, -1.0}, {2.0, 0.0}});
        assertComplexResult(real, complex, real.add(complex));
        assertComplexResult(complex, real, complex.add(real));
        assertComplexResult(complex, complex, complex.add(complex));
    }

    @Test
    public void edgeValuesAndOneByOneMatricesRemainBitwiseEquivalent() {
        Matrix a = matrix(1, 1, -0.0);
        Matrix b = matrix(1, 1, Double.POSITIVE_INFINITY);
        MatrixExpr expression = MatrixExpr.input(a).scale(-1.0).add(MatrixExpr.input(b));
        Matrix actual = compile(expression, FusionStrategy.GENERALIZED).execute();
        Matrix expected = a.multiplyScalar(-1.0).add(b);
        assertMatrixEquals(expected, actual);
        assertTrue(Double.isInfinite(actual.get(0, 0)));
    }

    @Test
    public void zeroSizedDenseInputsStillProduceTheCorrectFusedShape() {
        Matrix a = new Matrix(0, 3);
        Matrix b = new Matrix(0, 3);
        CompiledMatrixProgram program = compile(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)),
            FusionStrategy.GENERALIZED);
        assertTrue(program.cpuPlan().hasFusedRegionStep());
        Matrix result = program.execute();
        assertEquals(0, result.getRowCount());
        assertEquals(3, result.getColumnCount());
        assertEquals(0, result.getRawData().length);
    }

    @Test
    public void r1PlansOnlyTheFinalFusedValue() {
        Matrix a = matrix(8, 8, sequence(64));
        Matrix b = matrix(8, 8, sequence(64, 10));
        CompiledMatrixProgram program = compile(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)).scale(0.5),
            FusionStrategy.GENERALIZED);
        CpuExecutionPlan plan = program.cpuPlan();

        assertEquals(3, plan.logicalTemporaryCount());
        assertEquals(2, plan.fusionElidedTemporaryCount());
        assertEquals(1, plan.materializedTemporaryCount());
        assertEquals(1, plan.physicalSlotCount());
        for (LogicalBuffer buffer : plan.elidedTemporaryBuffers()) {
            assertFalse(plan.physicalMemoryPlan().hasSlot(buffer));
        }
        assertEquals(1, plan.fusionMetrics().fullMatrixPassesAfter());
    }

    @Test
    public void selectorOffMaterializesAndLegacyKeepsM4Pair() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b));

        CompiledMatrixProgram off = compile(expression, FusionStrategy.OFF);
        assertFalse(off.cpuPlan().hasFusedElementwiseStep());
        assertEquals(2, off.cpuPlan().stepCount());
        assertEquals(0, off.cpuPlan().elidedTemporaryCount());
        assertMatrixEquals(a.multiplyScalar(2.0).add(b), off.execute());

        CompiledMatrixProgram legacy = compile(expression, FusionStrategy.LEGACY);
        assertTrue(legacy.cpuPlan().steps().get(0) instanceof CpuFusedElementwiseStep);
        assertEquals(1, legacy.cpuPlan().elidedTemporaryCount());
        assertMatrixEquals(a.multiplyScalar(2.0).add(b), legacy.execute());
    }

    @Test
    public void generalizedFusionStopsAtGemmBoundary() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        Matrix c = matrix(2, 2, 2, 1, 4, 3);
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0)
            .matmul(MatrixExpr.input(b)).add(MatrixExpr.input(c));

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        assertEquals(3, program.cpuPlan().stepCount());
        assertEquals(CpuStepKind.ELEMENTWISE, program.cpuPlan().steps().get(0).kind());
        assertEquals(CpuStepKind.GEMM, program.cpuPlan().steps().get(1).kind());
        assertEquals(CpuStepKind.ELEMENTWISE, program.cpuPlan().steps().get(2).kind());
        assertFalse(program.cpuPlan().hasFusedRegionStep());
        assertMatrixEquals(
            Gemm.multiply(a.multiplyScalar(2.0), b).add(c),
            program.execute());
    }

    @Test
    public void escapingInternalProducerRejectsTheUnsafeMaximalRegion() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        Matrix c = matrix(2, 2, 2, 1, 4, 3);
        MatrixExpr shared = MatrixExpr.input(a).scale(2.0);
        MatrixExpr branch = shared.add(MatrixExpr.input(b));
        MatrixExpr expression = branch.add(shared.matmul(MatrixExpr.input(c)));

        CompiledMatrixProgram program = compile(expression, FusionStrategy.GENERALIZED);
        assertFalse(program.cpuPlan().hasFusedRegionStep());
        assertTrue(program.cpuPlan().fusionMetrics().rejectedRegionCount() > 0);
        assertTrue(program.cpuPlan().fusionMetrics().decisions().stream()
            .anyMatch(decision -> decision.contains("escaping producer")));
        assertMatrixEquals(
            a.multiplyScalar(2.0).add(b)
                .add(Gemm.multiply(a.multiplyScalar(2.0), c)),
            program.execute());
    }

    @Test
    public void unknownAliasFactsRejectAPlannerCandidate() {
        GeneralizedFusionPlanner.Result result = GeneralizedFusionPlanner.plan(
            unknownAliasSchedule());

        assertTrue(result.acceptedRegions().isEmpty());
        assertEquals(1, result.candidateRegionCount());
        assertEquals(1, result.rejectedRegionCount());
        assertTrue(result.decisions().stream()
            .anyMatch(decision -> decision.contains("UNKNOWN alias/dependence")));
    }

    @Test
    public void incompatibleProducerDomainRejectsFusionInsteadOfGuessing() {
        GeneralizedFusionPlanner.Result result = GeneralizedFusionPlanner.plan(
            incompatibleFusionSchedule());

        assertTrue(result.acceptedRegions().isEmpty());
        assertTrue(result.decisions().toString(), result.decisions().stream()
            .anyMatch(decision -> decision.contains("incompatible")));
    }

    @Test
    public void explicitStrategyOverloadDoesNotDependOnGlobalProperty() {
        String previous = System.getProperty(FusionStrategy.PROPERTY);
        try {
            System.setProperty(FusionStrategy.PROPERTY, "off");
            Matrix a = matrix(1, 2, 1, 2);
            Matrix b = matrix(1, 2, 3, 4);
            CompiledMatrixProgram program = compile(
                MatrixExpr.input(a).scale(4.0).add(MatrixExpr.input(b)),
                FusionStrategy.GENERALIZED);
            assertTrue(program.cpuPlan().hasFusedRegionStep());
            assertMatrixEquals(a.multiplyScalar(4.0).add(b), program.execute());
        } finally {
            if (previous == null) {
                System.clearProperty(FusionStrategy.PROPERTY);
            } else {
                System.setProperty(FusionStrategy.PROPERTY, previous);
            }
        }
    }

    private static CompiledMatrixProgram compile(MatrixExpr expression,
                                                  FusionStrategy strategy) {
        return MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(), strategy);
    }

    private static SchedulePlan unknownAliasSchedule() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        MatrixShape shape = new MatrixShape(2, 2);
        LogicalBuffer first = temporaryBuffer(0, "tmp0", shape);
        LogicalBuffer result = temporaryBuffer(1, "tmp1", shape);
        LogicalBuffer left = externalBuffer(2, "left", shape);
        LogicalBuffer aliasedWriter = externalBuffer(3, "writer", shape);
        IterationDomain domain = domain(i, j, 2, 2);

        AffineStatement firstAdd = new AffineStatement(
            0, StatementKind.ADD, domain,
            List.of(
                AffineAccess.write(first, expr(i), expr(j)),
                AffineAccess.read(left, expr(i), expr(j)),
                AffineAccess.read(aliasedWriter, expr(i), expr(j))),
            "%0[i,j] = %2[i,j] + %3[i,j]");
        AffineStatement secondAdd = new AffineStatement(
            1, StatementKind.ADD, domain,
            List.of(
                AffineAccess.write(result, expr(i), expr(j)),
                AffineAccess.read(first, expr(i), expr(j)),
                AffineAccess.read(left, expr(i), expr(j))),
            "%1[i,j] = %0[i,j] + %2[i,j]");
        AffineStatement outsideWriter = new AffineStatement(
            2, StatementKind.SYNTHETIC, domain,
            List.of(AffineAccess.write(aliasedWriter, expr(i), expr(j))),
            "%3[i,j] = outside");
        return SchedulePlan.initial(new AffineProgram(
            OptimizationSemantics.STRICT,
            List.of(first, result, left, aliasedWriter),
            List.of(i, j),
            List.of(firstAdd, secondAdd, outsideWriter),
            result,
            DependenceGraph.analyze(List.of(firstAdd, secondAdd, outsideWriter))));
    }

    private static SchedulePlan incompatibleFusionSchedule() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer first = temporaryBuffer(0, "tmp0", new MatrixShape(2, 2));
        LogicalBuffer result = temporaryBuffer(1, "tmp1", new MatrixShape(2, 3));
        LogicalBuffer input = externalBuffer(2, "input", new MatrixShape(2, 2));
        AffineStatement firstScale = new AffineStatement(
            0, StatementKind.SCALE, domain(i, j, 2, 2),
            List.of(
                AffineAccess.write(first, expr(i), expr(j)),
                AffineAccess.read(input, expr(i), expr(j))),
            "%0[i,j] = 2.0 * %2[i,j]");
        AffineStatement mismatchedScale = new AffineStatement(
            1, StatementKind.SCALE, domain(i, j, 2, 3),
            List.of(
                AffineAccess.write(result, expr(i), expr(j)),
                AffineAccess.read(first, expr(i), expr(j))),
            "%1[i,j] = 3.0 * %0[i,j]");
        return SchedulePlan.initial(new AffineProgram(
            OptimizationSemantics.STRICT,
            List.of(first, result, input),
            List.of(i, j),
            List.of(firstScale, mismatchedScale),
            result,
            DependenceGraph.analyze(List.of(firstScale, mismatchedScale))));
    }

    private static LogicalBuffer temporaryBuffer(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.TEMPORARY, BufferOwnership.OWNED,
            MemorySpace.UNKNOWN, shape);
    }

    private static LogicalBuffer externalBuffer(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.EXTERNAL_INPUT, BufferOwnership.BORROWED,
            MemorySpace.HEAP, shape);
    }

    private static IterationDomain domain(AffineVariable i,
                                          AffineVariable j,
                                          long rows,
                                          long columns) {
        return IterationDomain.of(
            IterationDomain.range(i, 0, rows), IterationDomain.range(j, 0, columns));
    }

    private static AffineExpr expr(AffineVariable variable) {
        return AffineExpr.variable(variable);
    }

    private static CpuFusedRegionStep.ExecutionPath executeAndPath(
        CompiledMatrixProgram program,
        CpuFusedRegionStep step) {
        program.execute();
        return step.lastExecutionPath();
    }

    private static Matrix matrix(int rows, int columns, double... values) {
        assertEquals(rows * columns, values.length);
        Matrix result = new Matrix(rows, columns);
        System.arraycopy(values, 0, result.getRawData(), 0, values.length);
        return result;
    }

    private static double[] sequence(int count) {
        return sequence(count, 1);
    }

    private static double[] sequence(int count, int offset) {
        double[] result = new double[count];
        for (int index = 0; index < result.length; index++) {
            result[index] = offset + index;
        }
        return result;
    }

    private static void assertComplexResult(Matrix left, Matrix right, Matrix expected) {
        CompiledMatrixProgram program = compile(
            MatrixExpr.input(left).add(MatrixExpr.input(right)),
            FusionStrategy.GENERALIZED);
        Matrix actual = program.execute();
        assertTrue(actual.hasImagData());
        assertMatrixEquals(expected, actual);
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int row = 0; row < expected.getRowCount(); row++) {
            for (int column = 0; column < expected.getColumnCount(); column++) {
                assertEquals(expected.get(row, column), actual.get(row, column), 0.0);
                assertEquals(expected.getImag(row, column), actual.getImag(row, column), 0.0);
            }
        }
    }
}
