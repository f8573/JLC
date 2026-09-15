package net.faulj.compiler.matrix.schedule;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

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
import net.faulj.compiler.matrix.affine.DependenceGraph;
import net.faulj.compiler.matrix.affine.DependenceKind;
import net.faulj.compiler.matrix.affine.DependenceStatus;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.StatementKind;

/**
 * Focused M3 coverage for immutable schedules and conservative legality.
 */
public class MatrixScheduleTest {
    @Test
    public void initialScheduleIsDeterministicAndFollowsDomainOrder() {
        AffineProgram program = matmulProgram(OptimizationSemantics.STRICT);
        SchedulePlan first = SchedulePlan.initial(program);
        SchedulePlan second = SchedulePlan.initial(program);

        assertEquals(first.dump(), second.dump());
        assertEquals(2, first.root().children().size());
        assertEquals(List.of("i", "j"), loopNames(first.region(0)));
        assertEquals(List.of("i", "j", "k"), loopNames(first.region(1)));
        assertTrue(first.history().isEmpty());
    }

    @Test
    public void adjacentInterchangeIsLegalWhenDirectionsRemainPositive() {
        SchedulePlan initial = SchedulePlan.initial(matmulProgram(OptimizationSemantics.STRICT));
        ScheduleTransformResult result = initial.interchange(1, "i", "j");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(List.of("j", "i", "k"), loopNames(result.schedule().region(1)));
        assertNotSame(initial, result.schedule());
    }

    @Test
    public void correctedOffsetDirectionAllowsPositiveInterchange() {
        AffineProgram program = offsetRawProgram();
        ScheduleRegion merged = new ScheduleRegion(
            program.statements(),
            new ScheduleBand(
                loops(program.statements().get(0)),
                new ScheduleSequence(program.statements().stream()
                    .map(ScheduleStatement::new).toList())));
        SchedulePlan schedule = SchedulePlan.of(program, List.of(merged));

        ScheduleTransformResult result = schedule.interchange(0, "i", "j");
        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(List.of("j", "i"), loopNames(result.schedule().region(0)));
    }

    @Test
    public void unknownDependenceRejectsInterchange() {
        AffineProgram program = ambiguousExternalProgram();
        SchedulePlan initial = SchedulePlan.initial(program);
        ScheduleRegion region = new ScheduleRegion(
            List.of(program.statements().get(0), program.statements().get(1)),
            new ScheduleBand(
                loops(program.statements().get(0)),
                new ScheduleSequence(List.of(
                    new ScheduleStatement(program.statements().get(0)),
                    new ScheduleStatement(program.statements().get(1))))));
        SchedulePlan mergedView = SchedulePlan.of(program, List.of(region));

        assertEquals(DependenceStatus.UNKNOWN,
            program.dependenceGraph().query(
                program.statements().get(0), program.statements().get(1), DependenceKind.RAW));
        ScheduleTransformResult result = mergedView.interchange(0, "i", "j");
        assertEquals(LegalityStatus.UNKNOWN, result.status());
        assertTrue(result.explanation().contains("UNKNOWN"));
        assertEquals(2, initial.regions().size());
    }

    @Test
    public void stripMineExactMultipleHasExactTileExtent() {
        SchedulePlan initial = SchedulePlan.initial(addProgram(8, 4));
        ScheduleTransformResult result = initial.stripMine(0, "i", 4);

        assertEquals(LegalityStatus.LEGAL, result.status());
        ScheduleLoop outer = result.schedule().region(0).band().loop(0);
        ScheduleLoop inner = result.schedule().region(0).band().loop(1);
        assertEquals("io", outer.variable().name());
        assertEquals(2L, outer.upperBound());
        assertEquals("ii", inner.variable().name());
        assertEquals(4L, inner.upperBound());
        assertTrue(inner.isBoundedByGuard());
    }

    @Test
    public void stripMineRemainderUsesCeilingTileCountAndGuard() {
        ScheduleTransformResult result = SchedulePlan.initial(addProgram(10, 4))
            .stripMine(0, "i", 4);

        assertEquals(LegalityStatus.LEGAL, result.status());
        ScheduleLoop outer = result.schedule().region(0).band().loop(0);
        ScheduleLoop inner = result.schedule().region(0).band().loop(1);
        assertEquals(3L, outer.upperBound());
        assertEquals(4L, inner.upperBound());
        assertTrue(inner.guard().contains("i < 10"));
        assertTrue(inner.valueExpression().toString().contains("io"));
    }

    @Test
    public void multidimensionalTilingComposesExplicitStripMines() {
        SchedulePlan initial = SchedulePlan.initial(addProgram(10, 9));
        ScheduleTransformResult first = initial.stripMine(0, "i", 4);
        ScheduleTransformResult second = first.schedule().stripMine(0, "j", 3);

        assertEquals(LegalityStatus.LEGAL, first.status());
        assertEquals(LegalityStatus.LEGAL, second.status());
        assertEquals(List.of("io", "ii", "jo", "ji"), loopNames(second.schedule().region(0)));
        assertEquals(3L, second.schedule().region(0).band().loop(2).upperBound());
    }

    @Test
    public void invalidTileSizeIsRejected() {
        ScheduleTransformResult zero = SchedulePlan.initial(addProgram(4, 4))
            .stripMine(0, "i", 0);
        ScheduleTransformResult negative = SchedulePlan.initial(addProgram(4, 4))
            .stripMine(0, "i", -2);

        assertEquals(LegalityStatus.ILLEGAL, zero.status());
        assertEquals(LegalityStatus.ILLEGAL, negative.status());
        assertTrue(zero.explanation().contains("positive"));
    }

    @Test
    public void identicalElementwiseDomainsCanBeFused() {
        MatrixExpr a = MatrixExpr.symbolicInput("A", new MatrixShape(3, 4));
        MatrixExpr b = MatrixExpr.symbolicInput("B", new MatrixShape(3, 4));
        AffineProgram program = lower(a.scale(2.0).add(b));
        SchedulePlan initial = SchedulePlan.initial(program);

        ScheduleTransformResult result = initial.fusion(0, 1);

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(1, result.schedule().regions().size());
        assertEquals(2, result.schedule().region(0).statements().size());
        assertTrue(result.schedule().region(0).band().body() instanceof ScheduleSequence);
    }

    @Test
    public void incompatibleDomainsRejectFusion() {
        AffineProgram program = incompatibleDomainProgram();
        ScheduleTransformResult result = SchedulePlan.initial(program).fusion(0, 1);

        assertEquals(LegalityStatus.ILLEGAL, result.status());
        assertTrue(result.explanation().contains("incompatible iteration domains"));
    }

    @Test
    public void reverseDependenceBlocksFusion() {
        AffineProgram program = sameTempProducerConsumerProgram();
        List<ScheduleRegion> reversed = List.of(
            singletonRegion(program.statements().get(1)),
            singletonRegion(program.statements().get(0)));
        ScheduleTransformResult result = SchedulePlan.of(program, reversed).fusion(1, 0);

        assertEquals(LegalityStatus.ILLEGAL, result.status());
        assertTrue(result.explanation().contains("would be reversed"));
    }

    @Test
    public void independentMatMulOuterLoopCanBeMarkedParallel() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.STRICT)).parallel(1, "i");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertTrue(result.schedule().region(1).band().loop(0)
            .hasAnnotation(ScheduleAnnotation.PARALLEL));
    }

    @Test
    public void dependenceCarriedLoopCannotBeMarkedParallel() {
        AffineProgram program = offsetRawProgram();
        ScheduleRegion merged = new ScheduleRegion(
            program.statements(),
            new ScheduleBand(
                loops(program.statements().get(0)),
                new ScheduleSequence(program.statements().stream()
                    .map(ScheduleStatement::new).toList())));
        ScheduleTransformResult result = SchedulePlan.of(program, List.of(merged))
            .parallel(0, "i");

        assertEquals(LegalityStatus.ILLEGAL, result.status());
        assertTrue(result.explanation().contains("carried by i"));
    }

    @Test
    public void strictReductionDimensionCannotBeMarkedParallel() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.STRICT)).parallel(1, "k");

        assertEquals(LegalityStatus.ILLEGAL, result.status());
        assertTrue(result.explanation().contains("STRICT reduction dimension"));
    }

    @Test
    public void relaxedReductionParallelEligibilityIsMetadataOnly() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.RELAXED)).parallel(1, "k");

        assertEquals(LegalityStatus.LEGAL, result.status());
        ScheduleLoop loop = result.schedule().region(1).band().loop(2);
        assertTrue(loop.hasAnnotation(ScheduleAnnotation.REDUCTION_PARALLEL_ELIGIBLE));
        assertFalse(loop.hasAnnotation(ScheduleAnnotation.PARALLEL));
        assertTrue(result.schedule().hasReductionReassociationMetadata());
    }

    @Test
    public void independentLoopCanBeMarkedVectorEligible() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.STRICT)).vector(1, "j");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertTrue(result.schedule().region(1).band().loop(1)
            .hasAnnotation(ScheduleAnnotation.VECTOR));
    }

    @Test
    public void strictReductionDimensionCannotBeMarkedVectorEligible() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.STRICT)).vector(1, "k");

        assertEquals(LegalityStatus.ILLEGAL, result.status());
    }

    @Test
    public void unknownDependenceRejectsVectorMarking() {
        AffineProgram program = ambiguousExternalProgram();
        ScheduleRegion region = new ScheduleRegion(
            List.of(program.statements().get(0), program.statements().get(1)),
            new ScheduleBand(
                loops(program.statements().get(0)),
                new ScheduleSequence(List.of(
                    new ScheduleStatement(program.statements().get(0)),
                    new ScheduleStatement(program.statements().get(1))))));

        ScheduleTransformResult result = SchedulePlan.of(program, List.of(region)).vector(0, "i");

        assertEquals(LegalityStatus.UNKNOWN, result.status());
        assertTrue(result.explanation().contains("UNKNOWN"));
    }

    @Test
    public void matMulUpdateScheduleIsAnExplicitIJKBand() {
        SchedulePlan schedule = SchedulePlan.initial(matmulProgram(OptimizationSemantics.STRICT));
        ScheduleRegion update = schedule.region(1);

        assertEquals(List.of("i", "j", "k"), loopNames(update));
        assertTrue(update.band().body() instanceof ScheduleStatement);
        assertEquals(1, ((ScheduleStatement) update.band().body()).id());
    }

    @Test
    public void strictReductionOrderingSurvivesNonReductionInterchange() {
        SchedulePlan initial = SchedulePlan.initial(matmulProgram(OptimizationSemantics.STRICT));
        ScheduleTransformResult result = initial.interchange(1, "i", "j");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(List.of("j", "i", "k"), loopNames(result.schedule().region(1)));
        assertFalse(result.schedule().hasReductionReassociationMetadata());
    }

    @Test
    public void relaxedReductionMayReorderWhenExplicitlyAllowed() {
        ScheduleTransformResult result = SchedulePlan.initial(
            matmulProgram(OptimizationSemantics.RELAXED)).interchange(1, "j", "k");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(List.of("i", "k", "j"), loopNames(result.schedule().region(1)));
        assertTrue(result.schedule().hasReductionReassociationMetadata());
        assertTrue(result.schedule().dump().contains("reduction-reassociation"));
    }

    @Test
    public void scheduleDumpAndHistoryAreDeterministic() {
        SchedulePlan first = SchedulePlan.initial(addProgram(5, 7));
        SchedulePlan second = SchedulePlan.initial(addProgram(5, 7));
        SchedulePlan firstTransformed = first.stripMine(0, "j", 3).schedule()
            .vector(0, "ji").schedule();
        SchedulePlan secondTransformed = second.stripMine(0, "j", 3).schedule()
            .vector(0, "ji").schedule();

        assertEquals(first.dump(), second.dump());
        assertEquals(firstTransformed.dump(), secondTransformed.dump());
        assertTrue(firstTransformed.dump().contains("transformation history:"));
        assertTrue(firstTransformed.dump().contains("strip-mine(j,3)"));
        assertTrue(firstTransformed.dump().contains("vector(ji)"));
    }

    @Test
    public void transformationHistoryIsImmutableAcrossScheduleBranches() {
        SchedulePlan initial = SchedulePlan.initial(addProgram(4, 4));
        SchedulePlan parallel = initial.parallel(0, "i").schedule();
        SchedulePlan vector = initial.vector(0, "j").schedule();

        assertTrue(initial.history().isEmpty());
        assertEquals(1, parallel.history().size());
        assertEquals(1, vector.history().size());
        assertTrue(parallel.history().get(0).operation().contains("parallel(i)"));
        assertTrue(vector.history().get(0).operation().contains("vector(j)"));
    }

    @Test
    public void candidateGenerationIsFiniteAndCappedAt32() {
        ScheduleCandidates candidates = ScheduleCandidateGenerator.generate(
            manyIndependentStatementsProgram(10));

        assertEquals(ScheduleCandidateGenerator.MAX_CANDIDATES, candidates.maximumAllowed());
        assertEquals(32, candidates.count());
        assertTrue(candidates.isWithinBound());
    }

    @Test
    public void transformationsDoNotMutateAffineProgram() {
        AffineProgram program = matmulProgram(OptimizationSemantics.RELAXED);
        String before = program.dump();
        SchedulePlan initial = SchedulePlan.initial(program);
        ScheduleTransformResult result = initial.interchange(1, "j", "k");

        assertEquals(LegalityStatus.LEGAL, result.status());
        assertEquals(before, program.dump());
        assertSame(program, result.schedule().program());
        assertNotSame(initial, result.schedule());
    }

    @Test
    public void originalDependenceGraphRemainsTheSameObjectAndDump() {
        AffineProgram program = matmulProgram(OptimizationSemantics.RELAXED);
        DependenceGraph graph = program.dependenceGraph();
        String before = graph.dump();
        SchedulePlan schedule = SchedulePlan.initial(program).interchange(1, "j", "k").schedule();

        assertSame(graph, schedule.dependenceGraph());
        assertEquals(before, graph.dump());
    }

    @Test
    public void m1PlannerBehaviorRemainsUnchanged() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(1000, 10))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(10, 1000)))
            .matmul(MatrixExpr.symbolicInput("C", new MatrixShape(1000, 10)));
        ExecutionPlan strict = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);
        ExecutionPlan relaxed = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);

        assertEquals(20_000_000L, strict.scalarMultiplicationCost());
        assertEquals(200_000L, relaxed.scalarMultiplicationCost());
    }

    @Test
    public void m2AffineFactsRemainVisibleAfterScheduleConstruction() {
        AffineProgram program = matmulProgram(OptimizationSemantics.STRICT);

        assertEquals(2, program.statements().size());
        assertEquals(StatementKind.MATMUL_INIT, program.statements().get(0).kind());
        assertEquals(StatementKind.MATMUL_UPDATE, program.statements().get(1).kind());
        assertEquals(AccessKind.REDUCTION,
            program.statements().get(1).accesses().get(2).kind());
        SchedulePlan schedule = SchedulePlan.initial(program).interchange(1, "i", "j").schedule();
        assertEquals("i in [0,2) && j in [0,2) && k in [0,3)",
            schedule.program().statements().get(1).domain().toString());
    }

    private static AffineProgram lower(MatrixExpr expression) {
        return net.faulj.compiler.matrix.affine.MatrixAffineCompiler.lower(
            MatrixCompiler.compile(expression));
    }

    private static AffineProgram matmulProgram(OptimizationSemantics semantics) {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 2)));
        return net.faulj.compiler.matrix.affine.MatrixAffineCompiler.lower(
            MatrixCompiler.compile(expression, semantics));
    }

    private static AffineProgram addProgram(int rows, int columns) {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(rows, columns))
            .add(MatrixExpr.symbolicInput("B", new MatrixShape(rows, columns)));
        return lower(expression);
    }

    private static List<String> loopNames(ScheduleRegion region) {
        return region.band().loops().stream()
            .map(loop -> loop.inductionVariable().name())
            .toList();
    }

    private static List<ScheduleLoop> loops(AffineStatement statement) {
        return statement.domain().ranges().stream()
            .map(range -> new ScheduleLoop(
                range.variable(), range.lowerInclusive(), range.upperExclusive(), 1L))
            .toList();
    }

    private static ScheduleRegion singletonRegion(AffineStatement statement) {
        return new ScheduleRegion(
            List.of(statement),
            new ScheduleBand(loops(statement), new ScheduleStatement(statement)));
    }

    private static AffineProgram offsetRawProgram() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer buffer = temporaryBuffer(0, "tmp", new MatrixShape(3, 3));
        IterationDomain domain = domain(i, j, 3, 3);
        AffineStatement producer = new AffineStatement(
            0,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.write(buffer, variable(i), variable(j))),
            "%0[i,j] = value");
        AffineStatement consumer = new AffineStatement(
            1,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.read(buffer, variable(i).add(1), variable(j).add(-1))),
            "use %0[i+1,j-1]");
        return program(List.of(buffer), List.of(producer, consumer), buffer,
            List.of(i, j), OptimizationSemantics.STRICT);
    }

    private static AffineProgram sameTempProducerConsumerProgram() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer buffer = temporaryBuffer(0, "tmp", new MatrixShape(2, 2));
        IterationDomain domain = domain(i, j, 2, 2);
        AffineStatement producer = new AffineStatement(
            0,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.write(buffer, variable(i), variable(j))),
            "%0[i,j] = value");
        AffineStatement consumer = new AffineStatement(
            1,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.read(buffer, variable(i), variable(j))),
            "use %0[i,j]");
        return program(List.of(buffer), List.of(producer, consumer), buffer,
            List.of(i, j), OptimizationSemantics.STRICT);
    }

    private static AffineProgram ambiguousExternalProgram() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer first = externalBuffer(0, "A");
        LogicalBuffer second = externalBuffer(1, "B");
        IterationDomain domain = domain(i, j, 2, 2);
        AffineStatement writer = new AffineStatement(
            0,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.write(first, variable(i), variable(j))),
            "%0[i,j] = value");
        AffineStatement reader = new AffineStatement(
            1,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.read(second, variable(i), variable(j))),
            "use %1[i,j]");
        return program(List.of(first, second), List.of(writer, reader), second,
            List.of(i, j), OptimizationSemantics.STRICT);
    }

    private static AffineProgram incompatibleDomainProgram() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        LogicalBuffer first = temporaryBuffer(0, "left", new MatrixShape(2, 2));
        LogicalBuffer second = temporaryBuffer(1, "right", new MatrixShape(2, 3));
        AffineStatement left = new AffineStatement(
            0,
            StatementKind.SYNTHETIC,
            domain(i, j, 2, 2),
            List.of(AffineAccess.write(first, variable(i), variable(j))),
            "%0[i,j] = value");
        AffineStatement right = new AffineStatement(
            1,
            StatementKind.SYNTHETIC,
            domain(i, j, 2, 3),
            List.of(AffineAccess.write(second, variable(i), variable(j))),
            "%1[i,j] = value");
        return program(List.of(first, second), List.of(left, right), second,
            List.of(i, j), OptimizationSemantics.STRICT);
    }

    private static AffineProgram manyIndependentStatementsProgram(int count) {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        IterationDomain domain = domain(i, j, 2, 2);
        List<LogicalBuffer> buffers = new ArrayList<>();
        List<AffineStatement> statements = new ArrayList<>();
        for (int index = 0; index < count; index++) {
            LogicalBuffer buffer = temporaryBuffer(index, "tmp" + index, new MatrixShape(2, 2));
            buffers.add(buffer);
            statements.add(new AffineStatement(
                index,
                StatementKind.SYNTHETIC,
                domain,
                List.of(AffineAccess.write(buffer, variable(i), variable(j))),
                "%" + index + "[i,j] = value"));
        }
        return program(buffers, statements, buffers.get(count - 1), List.of(i, j),
            OptimizationSemantics.STRICT);
    }

    private static AffineProgram program(List<LogicalBuffer> buffers,
                                         List<AffineStatement> statements,
                                         LogicalBuffer result,
                                         List<AffineVariable> variables,
                                         OptimizationSemantics semantics) {
        return new AffineProgram(
            semantics,
            buffers,
            variables,
            statements,
            result,
            DependenceGraph.analyze(statements));
    }

    private static IterationDomain domain(AffineVariable i,
                                         AffineVariable j,
                                         long rows,
                                         long columns) {
        return IterationDomain.of(
            IterationDomain.range(i, 0L, rows),
            IterationDomain.range(j, 0L, columns));
    }

    private static AffineExpr variable(AffineVariable variable) {
        return net.faulj.compiler.matrix.affine.AffineExpr.variable(variable);
    }

    private static LogicalBuffer temporaryBuffer(int id,
                                                 String name,
                                                 MatrixShape shape) {
        return new LogicalBuffer(
            id,
            name,
            BufferKind.TEMPORARY,
            BufferOwnership.OWNED,
            MemorySpace.UNKNOWN,
            shape);
    }

    private static LogicalBuffer externalBuffer(int id,
                                                String name) {
        return new LogicalBuffer(
            id,
            name,
            BufferKind.EXTERNAL_INPUT,
            BufferOwnership.BORROWED,
            MemorySpace.HEAP,
            new MatrixShape(2, 2));
    }
}
