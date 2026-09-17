package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.ExecutionPlan;
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
import net.faulj.compiler.matrix.affine.Dependence;
import net.faulj.compiler.matrix.affine.DependenceGraph;
import net.faulj.compiler.matrix.affine.DependenceKind;
import net.faulj.compiler.matrix.affine.DependenceStatus;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.ReductionMetadata;
import net.faulj.compiler.matrix.affine.ReductionSemantics;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.DependenceDirection;
import net.faulj.compiler.matrix.schedule.DependenceDirections;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;
import net.faulj.compiler.matrix.schedule.ScheduleStatement;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/** Counterexample-driven regression coverage for the post-M4 audit. */
public class MatrixCompilerAuditRegressionTest {
    @Test
    public void transposeThenScaleFusionIsUnknown() {
        AffineProgram program = AffineProgram.lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(2, 2)).transpose().scale(2.0)));

        assertEquals(LegalityStatus.UNKNOWN,
            SchedulePlan.initial(program).fusion(0, 1).status());
    }

    @Test
    public void pointwiseScaleAddFusionRemainsLegal() {
        AffineProgram program = AffineProgram.lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(3, 3)).scale(2.0)
                .add(MatrixExpr.symbolicInput("B", new MatrixShape(3, 3)))));
        assertEquals(LegalityStatus.LEGAL, SchedulePlan.initial(program).fusion(0, 1).status());
    }

    @Test
    public void offsetWawAndWarFusionCannotInterleave() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer state = temporary(0, "state", new MatrixShape(1, 4));
        IterationDomain points = domain(i, j, 1, 3);
        AffineStatement firstWrite = statement(0, points,
            List.of(AffineAccess.write(state, expr(i), expr(j))));
        AffineStatement shiftedWrite = statement(1, points,
            List.of(AffineAccess.write(state, expr(i), expr(j).add(1))));
        AffineProgram waw = program(List.of(state), List.of(firstWrite, shiftedWrite),
            state, List.of(i, j));
        assertEquals(LegalityStatus.UNKNOWN, SchedulePlan.initial(waw).fusion(0, 1).status());

        AffineStatement firstRead = statement(0, points,
            List.of(AffineAccess.read(state, expr(i), expr(j).add(1))));
        AffineStatement secondWrite = statement(1, points,
            List.of(AffineAccess.write(state, expr(i), expr(j))));
        AffineProgram war = program(List.of(state), List.of(firstRead, secondWrite),
            state, List.of(i, j));
        assertEquals(LegalityStatus.UNKNOWN, SchedulePlan.initial(war).fusion(0, 1).status());
    }

    @Test
    public void sameLocationPointwiseWriteReadFusionRemainsLegal() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer state = temporary(0, "state", new MatrixShape(2, 2));
        IterationDomain points = domain(i, j, 2, 2);
        AffineStatement write = statement(0, points,
            List.of(AffineAccess.write(state, expr(i), expr(j))));
        AffineStatement read = statement(1, points,
            List.of(AffineAccess.read(state, expr(i), expr(j))));
        AffineProgram program = program(List.of(state), List.of(write, read),
            state, List.of(i, j));
        assertEquals(LegalityStatus.LEGAL, SchedulePlan.initial(program).fusion(0, 1).status());
    }

    @Test
    public void identicalNoninjectiveWriteAccessBlocksParallelMarking() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer state = temporary(0, "state", new MatrixShape(4, 4));
        IterationDomain points = domain(i, j, 4, 4);
        for (AffineAccess access : List.of(
            AffineAccess.write(state, expr(i), AffineExpr.constant(0)),
            AffineAccess.readWrite(state, expr(i), AffineExpr.constant(0)))) {
            AffineStatement update = statement(0, points, List.of(access));
            AffineProgram program = program(List.of(state), List.of(update), state,
                List.of(i, j));
            assertEquals(DependenceStatus.UNKNOWN,
                program.dependenceGraph().query(update, update, DependenceKind.WAW));
            assertEquals(LegalityStatus.UNKNOWN,
                SchedulePlan.initial(program).parallel(0, "j").status());
        }
        AffineStatement canonical = statement(0, points,
            List.of(AffineAccess.write(state, expr(i), expr(j))));
        AffineProgram canonicalProgram = program(List.of(state), List.of(canonical), state,
            List.of(i, j));
        assertEquals(DependenceStatus.PROVEN_NONE,
            canonicalProgram.dependenceGraph().query(canonical, canonical, DependenceKind.WAW));
        assertEquals(LegalityStatus.LEGAL,
            SchedulePlan.initial(canonicalProgram).parallel(0, "j").status());
        AffineStatement collapsedRow = statement(0, points,
            List.of(AffineAccess.write(state, AffineExpr.constant(0), expr(j))));
        AffineProgram rowProgram = program(List.of(state), List.of(collapsedRow), state,
            List.of(i, j));
        assertEquals(LegalityStatus.UNKNOWN,
            SchedulePlan.initial(rowProgram).parallel(0, "i").status());
    }

    @Test
    public void ordinaryMixedAddCopiesImaginaryLaneExactly() {
        for (double imaginary : new double[]{-0.0, Double.NaN,
            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}) {
            Matrix real = scalar(2.0, null);
            Matrix complex = scalar(3.0, imaginary);
            assertOrdinaryAddMatchesEager(real, complex);
            assertOrdinaryAddMatchesEager(complex, real);
            assertCanonicalizedAddMatchesEager(real, complex);
            assertCanonicalizedAddMatchesEager(complex, real);
        }
        Matrix real = scalar(2.0, null);
        Matrix negativeZero = scalar(3.0, -0.0);
        assertEquals(0x8000000000000000L,
            Double.doubleToRawLongBits(MatrixCompiler.compileProgram(
                MatrixExpr.input(real).add(MatrixExpr.input(negativeZero)))
                .execute().getImag(0, 0)));
        assertEquals(0x8000000000000000L,
            Double.doubleToRawLongBits(MatrixCompiler.compileProgram(
                MatrixExpr.input(negativeZero).add(MatrixExpr.input(real)))
                .execute().getImag(0, 0)));
    }

    @Test
    public void shiftedLogicalDomainsRejectBeforeCpuExecution() {
        assertShiftedScaleRejected(1, 3, 0, 2);
        assertShiftedScaleRejected(0, 2, 1, 3);
        assertShiftedScaleRejected(-1, 1, 0, 2);
        assertShiftedScaleRejected(0, 2, -1, 1);
        assertCanonicalScaleAccepted(2, 2);
        assertCanonicalScaleAccepted(5, 3);
    }

    @Test
    public void generatedTileLoopsCannotBeRetiled() {
        SchedulePlan tiled = SchedulePlan.initial(addProgram(8, 8))
            .stripMine(0, "i", 4).schedule();

        assertEquals(LegalityStatus.ILLEGAL, tiled.stripMine(0, "ii", 2).status());
        assertEquals(LegalityStatus.ILLEGAL, tiled.stripMine(0, "io", 2).status());
    }

    @Test
    public void tileBinderCannotMoveAfterDependentInnerLoop() {
        SchedulePlan tiled = SchedulePlan.initial(addProgram(8, 8))
            .stripMine(0, "i", 4).schedule();

        assertEquals(LegalityStatus.ILLEGAL, tiled.interchange(0, "io", "ii").status());
    }

    @Test
    public void mayAliasReadWriteIsNotReadOnly() {
        LogicalBuffer left = external(0, "left", new MatrixShape(4, 4));
        LogicalBuffer right = external(1, "right", new MatrixShape(4, 4));
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        IterationDomain domain = domain(i, j, 4, 4);
        AffineStatement source = statement(0, domain,
            List.of(AffineAccess.readWrite(left, expr(i), expr(j))));
        AffineStatement sink = statement(1, domain,
            List.of(AffineAccess.read(right, expr(i), expr(j))));

        assertEquals(DependenceStatus.UNKNOWN,
            DependenceGraph.analyze(List.of(source, sink)).query(source, sink, DependenceKind.RAW));
    }

    @Test
    public void exactConflictDoesNotEraseAdditionalAliasUncertainty() {
        LogicalBuffer exact = temporary(0, "exact", new MatrixShape(4, 4));
        LogicalBuffer maybeLeft = external(1, "maybeLeft", new MatrixShape(4, 4));
        LogicalBuffer maybeRight = external(2, "maybeRight", new MatrixShape(4, 4));
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        IterationDomain domain = domain(i, j, 4, 4);
        AffineStatement source = statement(0, domain, List.of(
            AffineAccess.write(exact, expr(i), expr(j)),
            AffineAccess.write(maybeLeft, expr(i), expr(j))));
        AffineStatement sink = statement(1, domain, List.of(
            AffineAccess.read(exact, expr(i), expr(j)),
            AffineAccess.read(maybeRight, expr(i), expr(j))));

        assertEquals(DependenceStatus.UNKNOWN,
            DependenceGraph.analyze(List.of(source, sink)).query(source, sink, DependenceKind.RAW));
    }

    @Test
    public void constantOffsetDirectionsSolveSourceSinkEquality() {
        assertEquals(DependenceDirection.GREATER, offsetDirection(1));
        assertEquals(DependenceDirection.LESS, offsetDirection(-1));
        assertEquals(DependenceDirection.EQUAL, offsetDirection(0));
    }

    @Test
    public void unsupportedDirectionFormsRemainUnknown() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        assertEquals(DependenceDirection.UNKNOWN,
            customDirection(expr(i).scale(2), expr(i), i));
        assertEquals(DependenceDirection.UNKNOWN,
            customDirection(expr(i).add(expr(j)), expr(i).add(expr(j)), i));
        assertEquals(DependenceDirection.UNKNOWN,
            customDirection(expr(i), AffineExpr.constant(0), i));
    }

    @Test
    public void ordinarySelfRawWarAndWawAreAnalyzed() {
        LogicalBuffer buffer = temporary(0, "state", new MatrixShape(5, 1));
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        IterationDomain domain = domain(i, j, 3, 1);
        AffineStatement statement = statement(0, domain, List.of(
            AffineAccess.write(buffer, expr(i), expr(j)),
            AffineAccess.read(buffer, expr(i).add(1), expr(j)),
            AffineAccess.write(buffer, expr(i).add(2), expr(j))));
        DependenceGraph graph = DependenceGraph.analyze(List.of(statement));

        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(statement, statement, DependenceKind.RAW));
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(statement, statement, DependenceKind.WAR));
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(statement, statement, DependenceKind.WAW));
    }

    @Test
    public void reductionMetadataDoesNotSuppressUnrelatedSelfDependence() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        AffineVariable k = variable("k");
        LogicalBuffer output = temporary(0, "output", new MatrixShape(2, 2));
        LogicalBuffer state = temporary(1, "state", new MatrixShape(3, 2));
        IterationDomain domain = IterationDomain.of(
            IterationDomain.range(i, 0, 2), IterationDomain.range(j, 0, 2),
            IterationDomain.range(k, 0, 2));
        AffineStatement update = new AffineStatement(
            0, StatementKind.MATMUL_UPDATE, domain, List.of(
                AffineAccess.reduction(output, expr(i), expr(j)),
                AffineAccess.write(state, expr(i), expr(j)),
                AffineAccess.read(state, expr(i).add(1), expr(j))),
            "synthetic reduction with state",
            new ReductionMetadata(k, ReductionSemantics.REASSOCIATION_ELIGIBLE));
        DependenceGraph graph = DependenceGraph.analyze(List.of(update));

        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(update, update, DependenceKind.REDUCTION));
        assertEquals(DependenceStatus.UNKNOWN,
            graph.query(update, update, DependenceKind.RAW));
        assertEquals(DependenceStatus.UNKNOWN,
            graph.query(update, update, DependenceKind.WAW));
        AffineProgram program = program(List.of(output, state), List.of(update), output,
            List.of(i, j, k));
        assertFalse(SchedulePlan.initial(program).parallel(0, "k").accepted());
    }

    @Test
    public void fusedMixedRealComplexMatchesEagerForNonfiniteFactorsAndSignedZero() {
        for (double factor : new double[]{Double.NaN, Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY, -0.0}) {
            Matrix real = scalar(0.0, null);
            Matrix complex = scalar(1.0, -0.0);
            assertFusedMatchesEager(real, complex, factor, true);
            assertFusedMatchesEager(real, complex, factor, false);
            assertFusedMatchesEager(complex, real, factor, true);
            assertFusedMatchesEager(complex, real, factor, false);
        }
    }

    @Test
    public void cleanupClosesHiddenOwnedOffHeapMatrixExactlyOnce() {
        LogicalBuffer first = temporary(0, "first", new MatrixShape(1, 1));
        LogicalBuffer alias = temporary(1, "alias", new MatrixShape(1, 1));
        OffHeapMatrix hidden = new OffHeapMatrix(1, 1);
        CpuExecutionContext context = new CpuExecutionContext();
        context.bindOwned(first, hidden);
        context.bindOwned(alias, hidden);

        context.closeOwnedExcept(null, null);

        assertFalse(hidden.segment().scope().isAlive());
    }

    @Test
    public void cleanupRunsOnFailureWithoutClosingBorrowedInput() {
        LogicalBuffer owned = temporary(0, "owned", new MatrixShape(1, 1));
        LogicalBuffer borrowed = external(1, "borrowed", new MatrixShape(1, 1));
        OffHeapMatrix hidden = new OffHeapMatrix(1, 1);
        OffHeapMatrix input = new OffHeapMatrix(1, 1);
        try {
            CpuExecutionContext context = new CpuExecutionContext();
            context.bindOwned(owned, hidden);
            context.bind(borrowed, input);
            context.closeOwnedExcept(null, new IllegalStateException("execution failed"));

            assertFalse(hidden.segment().scope().isAlive());
            assertTrue(input.segment().scope().isAlive());
        } finally {
            input.close();
        }
    }

    @Test
    public void returnedOffHeapResultTransfersToCaller() {
        OffHeapMatrix left = new OffHeapMatrix(1, 1);
        OffHeapMatrix right = new OffHeapMatrix(1, 1);
        try {
            left.set(0, 0, 2.0);
            right.set(0, 0, 3.0);
            Matrix result = MatrixCompiler.compileProgram(
                MatrixExpr.input(left).matmul(MatrixExpr.input(right))).execute();
            try {
                assertTrue(result instanceof OffHeapMatrix);
                assertTrue(((OffHeapMatrix) result).segment().scope().isAlive());
                assertEquals(6.0, result.get(0, 0), 0.0);
            } finally {
                ((OffHeapMatrix) result).close();
            }
            assertTrue(left.segment().scope().isAlive());
            assertTrue(right.segment().scope().isAlive());
        } finally {
            left.close();
            right.close();
        }
    }

    @Test
    public void mismatchedM1M2ProvenanceIsRejected() {
        Matrix input = scalar(2.0, null);
        ExecutionPlan twice = MatrixCompiler.compile(MatrixExpr.input(input).scale(2.0));
        ExecutionPlan thrice = MatrixCompiler.compile(MatrixExpr.input(input).scale(3.0));
        SchedulePlan schedule = SchedulePlan.initial(AffineProgram.lower(twice));

        expectLoweringRejected(schedule, thrice, "provenance");
    }

    @Test
    public void borrowedOutputWriteIsRejected() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer output = external(0, "output", new MatrixShape(2, 2));
        LogicalBuffer left = external(1, "left", new MatrixShape(2, 2));
        LogicalBuffer right = external(2, "right", new MatrixShape(2, 2));
        AffineStatement add = new AffineStatement(0, StatementKind.ADD, domain(i, j, 2, 2),
            List.of(AffineAccess.write(output, expr(i), expr(j)),
                AffineAccess.read(left, expr(i), expr(j)),
                AffineAccess.read(right, expr(i), expr(j))), "borrowed output write");
        AffineProgram program = program(List.of(output, left, right), List.of(add), output,
            List.of(i, j));

        expectLoweringRejected(SchedulePlan.initial(program), null, "borrowed external");
    }

    @Test
    public void incompleteScheduleBandIsRejected() {
        AffineProgram program = addProgram(2, 2);
        AffineStatement statement = program.statements().get(0);
        ScheduleRegion truncated = new ScheduleRegion(List.of(statement),
            new ScheduleBand(List.of(new ScheduleLoop(variable("i"), 0, 2)),
                new ScheduleStatement(statement)));

        expectLoweringRejected(SchedulePlan.of(program, List.of(truncated)), null,
            "bind each domain variable");
    }

    @Test
    public void unsupportedNestedScheduleBodyIsRejected() {
        AffineProgram program = addProgram(2, 2);
        AffineStatement statement = program.statements().get(0);
        ScheduleBand nested = new ScheduleBand(List.of(new ScheduleLoop(variable("j"), 0, 2)),
            new ScheduleStatement(statement));
        ScheduleRegion region = new ScheduleRegion(List.of(statement),
            new ScheduleBand(List.of(new ScheduleLoop(variable("i"), 0, 2)), nested));

        expectLoweringRejected(SchedulePlan.of(program, List.of(region)), null, "nested");
    }

    @Test
    public void foreignAccessBufferIsRejected() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer output = temporary(0, "output", new MatrixShape(2, 2));
        LogicalBuffer foreign = external(1, "foreign", new MatrixShape(2, 2));
        AffineStatement scale = new AffineStatement(0, StatementKind.SCALE, domain(i, j, 2, 2),
            List.of(AffineAccess.write(output, expr(i), expr(j)),
                AffineAccess.read(foreign, expr(i), expr(j))), "foreign read");
        AffineProgram program = program(List.of(output), List.of(scale), output, List.of(i, j));

        expectLoweringRejected(SchedulePlan.initial(program), null, "foreign logical buffer");
    }

    @Test
    public void incompleteOutputCoverageIsRejected() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer output = temporary(0, "output", new MatrixShape(3, 2));
        LogicalBuffer input = external(1, "input", new MatrixShape(3, 2));
        AffineStatement scale = new AffineStatement(0, StatementKind.SCALE, domain(i, j, 2, 2),
            List.of(AffineAccess.write(output, expr(i), expr(j)),
                AffineAccess.read(input, expr(i), expr(j))), "partial scale");
        AffineProgram program = program(List.of(output, input), List.of(scale), output, List.of(i, j));

        expectLoweringRejected(SchedulePlan.initial(program), null, "exact canonical");
    }

    @Test
    public void emptyAndDegenerateReductionDomainsDoNotInventDependences() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        assertTrue(domain(i, j, 0, 3).isEmpty());
        assertNoReductionDependence(0);
        assertNoReductionDependence(1);
        assertHasReductionDependence(2);
    }

    @Test
    public void loopExecutorObservesInterchangedIterationOrder() {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        LogicalBuffer buffer = temporary(0, "output", new MatrixShape(2, 3));
        AffineStatement statement = statement(0, domain(i, j, 2, 3),
            List.of(AffineAccess.write(buffer, expr(i), expr(j))));
        ScheduleBand band = new ScheduleBand(
            List.of(new ScheduleLoop(j, 0, 3), new ScheduleLoop(i, 0, 2)),
            new ScheduleStatement(statement));
        List<String> observed = new ArrayList<>();

        CpuLoopExecutor.forEachPoint(band,
            bindings -> observed.add(bindings.get(i) + "," + bindings.get(j)));

        assertEquals(List.of("0,0", "1,0", "0,1", "1,1", "0,2", "1,2"), observed);
    }

    private static void assertFusedMatchesEager(Matrix scaled,
                                                Matrix addend,
                                                double factor,
                                                boolean scaledFirst) {
        MatrixExpr scaledExpression = MatrixExpr.input(scaled).scale(factor);
        MatrixExpr expression = scaledFirst
            ? scaledExpression.add(MatrixExpr.input(addend))
            : MatrixExpr.input(addend).add(scaledExpression);
        Matrix expected = scaledFirst
            ? scaled.multiplyScalar(factor).add(addend)
            : addend.add(scaled.multiplyScalar(factor));
        net.faulj.compiler.matrix.CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(expression);
        assertTrue(compiled.cpuPlan().hasFusedElementwiseStep());
        assertMatrixBitsEqual(expected, compiled.execute());
    }

    private static void assertOrdinaryAddMatchesEager(Matrix left, Matrix right) {
        Matrix expected = left.add(right);
        Matrix actual = MatrixCompiler.compileProgram(
            MatrixExpr.input(left).add(MatrixExpr.input(right))).execute();
        assertMatrixBitsEqual(expected, actual);
    }

    private static void assertCanonicalizedAddMatchesEager(Matrix left, Matrix right) {
        Matrix expected = left.multiplyScalar(1.0).add(right);
        Matrix actual = MatrixCompiler.compileProgram(
            MatrixExpr.input(left).scale(1.0).add(MatrixExpr.input(right))).execute();
        assertMatrixBitsEqual(expected, actual);
        Matrix expectedReversed = right.add(left.multiplyScalar(1.0));
        Matrix actualReversed = MatrixCompiler.compileProgram(
            MatrixExpr.input(right).add(MatrixExpr.input(left).scale(1.0))).execute();
        assertMatrixBitsEqual(expectedReversed, actualReversed);
    }

    private static void assertShiftedScaleRejected(long iLower, long iUpper,
                                                   long jLower, long jUpper) {
        expectLoweringRejected(scaleProgram(iLower, iUpper, jLower, jUpper), null,
            "exact canonical");
    }

    private static void assertCanonicalScaleAccepted(int rows, int columns) {
        SchedulePlan schedule = scaleProgram(0, rows, 0, columns);
        CpuExecutableSubsetValidator.validate(schedule);
        CpuExecutableSubsetValidator.validate(schedule.stripMine(0, "i", 2).schedule());
        CpuExecutableSubsetValidator.validate(schedule.stripMine(0, "i", 3).schedule());
    }

    private static SchedulePlan scaleProgram(long iLower, long iUpper,
                                             long jLower, long jUpper) {
        AffineVariable i = variable("i");
        AffineVariable j = variable("j");
        MatrixShape shape = new MatrixShape((int) (iUpper - iLower), (int) (jUpper - jLower));
        LogicalBuffer output = temporary(0, "output", shape);
        LogicalBuffer input = external(1, "input", shape);
        IterationDomain points = IterationDomain.of(
            IterationDomain.range(i, iLower, iUpper),
            IterationDomain.range(j, jLower, jUpper));
        AffineStatement scale = new AffineStatement(0, StatementKind.SCALE, points,
            List.of(AffineAccess.write(output, expr(i), expr(j)),
                AffineAccess.read(input, expr(i), expr(j))), "scale domain probe");
        return SchedulePlan.initial(program(List.of(output, input), List.of(scale), output,
            List.of(i, j)));
    }

    private static void assertMatrixBitsEqual(Matrix expected, Matrix actual) {
        assertEquals(expected.hasImagData(), actual.hasImagData());
        assertDoubleBits(expected.get(0, 0), actual.get(0, 0));
        assertDoubleBits(expected.getImag(0, 0), actual.getImag(0, 0));
    }

    private static void assertDoubleBits(double expected, double actual) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual));
        } else {
            assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
        }
    }

    private static DependenceDirection offsetDirection(long sinkOffset) {
        AffineVariable i = variable("i");
        return customDirection(expr(i), expr(i).add(sinkOffset), i);
    }

    private static DependenceDirection customDirection(AffineExpr sourceIndex,
                                                       AffineExpr sinkIndex,
                                                       AffineVariable variable) {
        AffineVariable j = variable("j");
        LogicalBuffer buffer = temporary(0, "buffer", new MatrixShape(8, 8));
        IterationDomain domain = domain(variable, j, 4, 4);
        AffineStatement source = statement(0, domain,
            List.of(AffineAccess.write(buffer, sourceIndex, expr(j))));
        AffineStatement sink = statement(1, domain,
            List.of(AffineAccess.read(buffer, sinkIndex, expr(j))));
        Dependence dependence = DependenceGraph.analyze(List.of(source, sink))
            .dependence(source, sink, DependenceKind.RAW);
        return DependenceDirections.analyze(dependence).direction(variable);
    }

    private static void assertNoReductionDependence(int kExtent) {
        AffineProgram program = matmulProgram(kExtent);
        AffineStatement update = program.statements().get(1);
        assertEquals(DependenceStatus.PROVEN_NONE,
            program.dependenceGraph().query(update, update, DependenceKind.REDUCTION));
    }

    private static void assertHasReductionDependence(int kExtent) {
        AffineProgram program = matmulProgram(kExtent);
        AffineStatement update = program.statements().get(1);
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            program.dependenceGraph().query(update, update, DependenceKind.REDUCTION));
    }

    private static AffineProgram matmulProgram(int kExtent) {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, kExtent))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(kExtent, 3)));
        return AffineProgram.lower(MatrixCompiler.compile(expression));
    }

    private static void expectLoweringRejected(SchedulePlan schedule,
                                               ExecutionPlan plan,
                                               String message) {
        try {
            if (plan == null) {
                CpuLowerer.lower(schedule);
            } else {
                CpuLowerer.lower(schedule, plan);
            }
            fail("Expected CPU lowering rejection");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(message));
        }
    }

    private static AffineProgram addProgram(int rows, int columns) {
        return AffineProgram.lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(rows, columns))
                .add(MatrixExpr.symbolicInput("B", new MatrixShape(rows, columns)))));
    }

    private static AffineProgram program(List<LogicalBuffer> buffers,
                                         List<AffineStatement> statements,
                                         LogicalBuffer result,
                                         List<AffineVariable> variables) {
        return new AffineProgram(OptimizationSemantics.STRICT, buffers, variables, statements,
            result, DependenceGraph.analyze(statements));
    }

    private static AffineStatement statement(int id,
                                             IterationDomain domain,
                                             List<AffineAccess> accesses) {
        return new AffineStatement(id, StatementKind.SYNTHETIC, domain, accesses,
            "audit probe S" + id);
    }

    private static IterationDomain domain(AffineVariable i,
                                          AffineVariable j,
                                          long rows,
                                          long columns) {
        return IterationDomain.of(
            IterationDomain.range(i, 0, rows), IterationDomain.range(j, 0, columns));
    }

    private static Matrix scalar(double real, Double imaginary) {
        Matrix matrix = new Matrix(1, 1);
        if (imaginary == null) {
            matrix.set(0, 0, real);
        } else {
            matrix.setComplex(0, 0, real, imaginary);
            if (!matrix.hasImagData()) {
                matrix.ensureImagData()[0] = imaginary;
            }
        }
        return matrix;
    }

    private static AffineVariable variable(String name) {
        return new AffineVariable(name);
    }

    private static AffineExpr expr(AffineVariable variable) {
        return AffineExpr.variable(variable);
    }

    private static LogicalBuffer temporary(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.TEMPORARY, BufferOwnership.OWNED,
            MemorySpace.UNKNOWN, shape);
    }

    private static LogicalBuffer external(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.EXTERNAL_INPUT, BufferOwnership.BORROWED,
            MemorySpace.HEAP, shape);
    }
}
