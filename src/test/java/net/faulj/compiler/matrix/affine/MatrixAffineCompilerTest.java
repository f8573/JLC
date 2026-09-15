package net.faulj.compiler.matrix.affine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * Focused M2 coverage for logical buffers, affine lowering, liveness, and
 * conservative dependence facts.
 */
public class MatrixAffineCompilerTest {
    @Test
    public void bufferAndStatementIdsAndDumpsAreDeterministic() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 4)))
            .transpose();

        AffineProgram first = MatrixAffineCompiler.lower(
            MatrixCompiler.compile(expression, OptimizationSemantics.STRICT));
        AffineProgram second = MatrixAffineCompiler.lower(
            MatrixCompiler.compile(expression, OptimizationSemantics.STRICT));

        assertEquals(first.dump(), second.dump());
        assertEquals(4, first.buffers().size());
        assertEquals(3, first.statements().size());
        for (int i = 0; i < first.buffers().size(); i++) {
            assertEquals(i, first.buffers().get(i).id());
        }
        for (int i = 0; i < first.statements().size(); i++) {
            assertEquals(i, first.statements().get(i).id());
        }
        assertEquals("i", first.variables().get(0).name());
        assertEquals("j", first.variables().get(1).name());
        assertEquals("k", first.variables().get(2).name());
    }

    @Test
    public void classifiesExternalHeapAndOffHeapInputsConservatively() {
        Matrix heap = new Matrix(2, 2);
        try (OffHeapMatrix offHeap = new OffHeapMatrix(2, 2)) {
            AffineProgram program = lower(
                MatrixExpr.input("heap", heap).matmul(MatrixExpr.input("offHeap", offHeap)));

            LogicalBuffer heapBuffer = buffer(program, "heap");
            LogicalBuffer offHeapBuffer = buffer(program, "offHeap");
            LogicalBuffer temporary = temporary(program, 0);

            assertEquals(BufferKind.EXTERNAL_INPUT, heapBuffer.kind());
            assertEquals(BufferOwnership.BORROWED, heapBuffer.ownership());
            assertEquals(MemorySpace.HEAP, heapBuffer.memorySpace());
            assertTrue(heapBuffer.lifetime().liveForEvaluation());

            assertEquals(BufferKind.EXTERNAL_INPUT, offHeapBuffer.kind());
            assertEquals(BufferOwnership.BORROWED, offHeapBuffer.ownership());
            assertEquals(MemorySpace.OFF_HEAP, offHeapBuffer.memorySpace());
            assertTrue(offHeapBuffer.lifetime().liveForEvaluation());

            assertEquals(BufferKind.TEMPORARY, temporary.kind());
            assertEquals(BufferOwnership.OWNED, temporary.ownership());
            assertEquals(MemorySpace.UNKNOWN, temporary.memorySpace());
        }
    }

    @Test
    public void symbolicBuffersHaveNoRuntimeOwnershipOrLifetime() {
        AffineProgram program = lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))));

        LogicalBuffer symbolic = program.buffers().get(0);
        assertEquals(BufferKind.SYMBOLIC, symbolic.kind());
        assertEquals(BufferOwnership.NONE, symbolic.ownership());
        assertEquals(MemorySpace.UNKNOWN, symbolic.memorySpace());
        assertFalse(symbolic.lifetime().hasRuntimeLifetime());
        assertNull(symbolic.lifetime().producer());
        assertTrue(program.statements().isEmpty());
    }

    @Test
    public void sameRuntimeMatrixIdentityMustAliasAndIsOneExternalBuffer() {
        Matrix shared = new Matrix(2, 2);
        MatrixExpr expression = MatrixExpr.input("left", shared)
            .add(MatrixExpr.input("right", shared));

        AffineProgram program = lower(expression);
        long externalCount = program.buffers().stream()
            .filter(LogicalBuffer::isExternalInput)
            .count();
        assertEquals(1L, externalCount);
        LogicalBuffer external = program.buffers().stream()
            .filter(LogicalBuffer::isExternalInput)
            .findFirst()
            .orElseThrow();
        assertEquals(AliasRelation.MUST_ALIAS, AliasAnalysis.between(external, external));
    }

    @Test
    public void temporaryAndExternalAliasRulesAreExplicit() {
        MatrixExpr left = MatrixExpr.symbolicInput("A", new MatrixShape(2, 2));
        MatrixExpr right = MatrixExpr.symbolicInput("B", new MatrixShape(2, 2));
        MatrixExpr otherLeft = MatrixExpr.symbolicInput("C", new MatrixShape(2, 2));
        MatrixExpr otherRight = MatrixExpr.symbolicInput("D", new MatrixShape(2, 2));
        AffineProgram program = lower(MatrixCompiler.compile(
            left.matmul(right).add(otherLeft.matmul(otherRight))));
        LogicalBuffer temporary = temporary(program, 0);
        LogicalBuffer secondTemporary = temporary(program, 1);
        LogicalBuffer firstSymbolic = program.buffers().get(0);
        LogicalBuffer secondSymbolic = program.buffers().get(1);

        assertEquals(AliasRelation.NO_ALIAS, AliasAnalysis.between(temporary, firstSymbolic));
        assertEquals(AliasRelation.MUST_ALIAS, AliasAnalysis.between(temporary, temporary));
        assertEquals(AliasRelation.NO_ALIAS, AliasAnalysis.between(temporary, secondTemporary));
        assertEquals(AliasRelation.MAY_ALIAS,
            AliasAnalysis.between(externalBuffer(0), externalBuffer(1)));
        assertEquals(AliasRelation.NO_ALIAS,
            AliasAnalysis.between(firstSymbolic, secondSymbolic));
    }

    @Test
    public void affineExpressionsRemainIntegerLinear() {
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        AffineExpr expression = AffineExpr.variable(i)
            .scale(2L)
            .add(AffineExpr.variable(j))
            .add(4L);

        assertEquals("2*i + j + 4", expression.toString());
        assertEquals(2L, expression.coefficient(i));
        assertEquals(1L, expression.coefficient(j));
        assertEquals(4L, expression.constant());
        assertFalse(expression.isConstant());
    }

    @Test
    public void lowersAddWithRectangularDomainAndAccesses() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .add(MatrixExpr.symbolicInput("B", new MatrixShape(2, 3)));
        AffineProgram program = lower(expression);

        AffineStatement statement = onlyStatement(program);
        assertEquals(StatementKind.ADD, statement.kind());
        assertEquals("i in [0,2) && j in [0,3)", statement.domain().toString());
        assertEquals(3, statement.accesses().size());
        assertEquals(AccessKind.WRITE, statement.accesses().get(0).kind());
        assertEquals(AccessKind.READ, statement.accesses().get(1).kind());
        assertEquals(AccessKind.READ, statement.accesses().get(2).kind());
        assertEquals("i", statement.accesses().get(1).index(0).toString());
        assertEquals("j", statement.accesses().get(1).index(1).toString());
        assertTrue(statement.computation().contains(" + "));
    }

    @Test
    public void lowersScaleWithScalarAndAccesses() {
        AffineProgram program = lower(MatrixExpr.symbolicInput("A", new MatrixShape(3, 2)).scale(2.5));
        AffineStatement statement = onlyStatement(program);

        assertEquals(StatementKind.SCALE, statement.kind());
        assertEquals("i in [0,3) && j in [0,2)", statement.domain().toString());
        assertEquals(AccessKind.WRITE, statement.accesses().get(0).kind());
        assertEquals(AccessKind.READ, statement.accesses().get(1).kind());
        assertTrue(statement.computation().contains("2.5"));
        assertEquals(statement.accesses().get(0).buffer(), program.resultBuffer());
    }

    @Test
    public void lowersTransposeWithPermutedAccess() {
        AffineProgram program = lower(MatrixExpr.symbolicInput("A", new MatrixShape(2, 3)).transpose());
        AffineStatement statement = onlyStatement(program);

        assertEquals(StatementKind.TRANSPOSE, statement.kind());
        assertEquals("i in [0,2) && j in [0,3)", statement.domain().toString());
        AffineAccess write = statement.accesses().get(0);
        AffineAccess read = statement.accesses().get(1);
        assertEquals("j", write.index(0).toString());
        assertEquals("i", write.index(1).toString());
        assertEquals("i", read.index(0).toString());
        assertEquals("j", read.index(1).toString());
    }

    @Test
    public void lowersMatMulIntoInitAndReductionUpdate() {
        AffineProgram program = lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
                .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 4))),
            OptimizationSemantics.STRICT));

        assertEquals(2, program.statements().size());
        AffineStatement init = program.statements().get(0);
        AffineStatement update = program.statements().get(1);
        assertEquals(StatementKind.MATMUL_INIT, init.kind());
        assertEquals("i in [0,2) && j in [0,4)", init.domain().toString());
        assertEquals(AccessKind.WRITE, init.accesses().get(0).kind());

        assertEquals(StatementKind.MATMUL_UPDATE, update.kind());
        assertEquals("i in [0,2) && j in [0,4) && k in [0,3)", update.domain().toString());
        assertEquals(AccessKind.READ, update.accesses().get(0).kind());
        assertEquals(AccessKind.READ, update.accesses().get(1).kind());
        assertEquals(AccessKind.REDUCTION, update.accesses().get(2).kind());
        assertEquals("[i,k]", bracketedIndices(update.accesses().get(0)));
        assertEquals("[k,j]", bracketedIndices(update.accesses().get(1)));
        assertEquals("[i,j]", bracketedIndices(update.accesses().get(2)));
        assertTrue(update.computation().contains("+= "));
        assertEquals("k", update.reduction().variable().name());
        assertEquals(ReductionSemantics.STRICT_ORDERED, update.reduction().semantics());
    }

    @Test
    public void producerConsumerAndReductionDependencesAreExplicit() {
        AffineProgram program = lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
                .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 2)))
                .transpose()));
        AffineStatement init = program.statements().get(0);
        AffineStatement update = program.statements().get(1);
        AffineStatement transpose = program.statements().get(2);
        DependenceGraph graph = program.dependenceGraph();

        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(init, update, DependenceKind.RAW));
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(init, update, DependenceKind.WAW));
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(update, update, DependenceKind.REDUCTION));
        assertEquals(DependenceStatus.PROVEN_DEPENDENCE,
            graph.query(update, transpose, DependenceKind.RAW));
        assertTrue(graph.dump().contains("REDUCTION"));
        assertTrue(graph.dump().contains("k -> k+1"));
    }

    @Test
    public void temporaryLifetimeEndsAfterFinalConsumerRead() {
        AffineProgram program = lower(MatrixCompiler.compile(
            MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
                .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 4)))
                .transpose()));
        LogicalBuffer product = temporary(program, 0);

        assertEquals(Integer.valueOf(0), product.lifetime().producer());
        assertEquals(Integer.valueOf(1), product.lifetime().firstUse());
        assertEquals(Integer.valueOf(2), product.lifetime().lastUse());
        assertEquals(Integer.valueOf(2), product.lifetime().lastUseStatementId());
        assertEquals(Integer.valueOf(2), program.resultBuffer().lifetime().producer());
    }

    @Test
    public void sharedDagProducerUsesOneTemporaryAndOneProducer() {
        MatrixExpr a = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3));
        MatrixExpr b = MatrixExpr.symbolicInput("B", new MatrixShape(3, 2));
        MatrixExpr shared = a.matmul(b);
        AffineProgram program = lower(MatrixCompiler.compile(shared.add(shared)));

        long temporaryCount = program.buffers().stream().filter(LogicalBuffer::isTemporary).count();
        assertEquals(2L, temporaryCount);
        assertEquals(3, program.statements().size());
        AffineStatement add = program.statements().get(2);
        assertSame(add.accesses().get(1).buffer(), add.accesses().get(2).buffer());
        assertEquals(1L, program.dependenceGraph().dependences().stream()
            .filter(edge -> edge.kind() == DependenceKind.REDUCTION).count());
        assertEquals(1L, countStatementsOfKind(program, StatementKind.MATMUL_INIT));
        assertEquals(1L, countStatementsOfKind(program, StatementKind.MATMUL_UPDATE));
    }

    @Test
    public void ambiguousExternalAliasProducesUnknownInsteadOfNone() {
        LogicalBuffer writerBuffer = externalBuffer(0);
        LogicalBuffer readerBuffer = externalBuffer(1);
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        IterationDomain domain = IterationDomain.of(
            IterationDomain.range(i, 0, 2), IterationDomain.range(j, 0, 2));
        AffineStatement writer = new AffineStatement(
            0,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.write(writerBuffer, AffineExpr.variable(i), AffineExpr.variable(j))),
            "%0[i,j] = value");
        AffineStatement reader = new AffineStatement(
            1,
            StatementKind.SYNTHETIC,
            domain,
            List.of(AffineAccess.read(readerBuffer, AffineExpr.variable(i), AffineExpr.variable(j))),
            "use %1[i,j]");

        DependenceGraph graph = DependenceGraph.analyze(List.of(writer, reader));

        assertEquals(AliasRelation.MAY_ALIAS,
            AliasAnalysis.between(writerBuffer, readerBuffer));
        assertEquals(DependenceStatus.UNKNOWN,
            graph.query(writer, reader, DependenceKind.RAW));
        assertTrue(graph.dump().contains("UNKNOWN"));
    }

    @Test
    public void strictAndRelaxedReductionMetadataDifferWithoutTransforming() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(4, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 5)));
        AffineProgram strict = lower(MatrixCompiler.compile(expression, OptimizationSemantics.STRICT));
        AffineProgram relaxed = lower(MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED));

        Dependence strictEdge = strict.dependenceGraph().dependence(
            strict.statements().get(1), strict.statements().get(1), DependenceKind.REDUCTION);
        Dependence relaxedEdge = relaxed.dependenceGraph().dependence(
            relaxed.statements().get(1), relaxed.statements().get(1), DependenceKind.REDUCTION);
        assertNotNull(strictEdge);
        assertNotNull(relaxedEdge);
        assertEquals(ReductionSemantics.STRICT_ORDERED, strictEdge.reduction().semantics());
        assertEquals(ReductionSemantics.REASSOCIATION_ELIGIBLE, relaxedEdge.reduction().semantics());
        assertFalse(strictEdge.reduction().allowsReassociation());
        assertTrue(relaxedEdge.reduction().allowsReassociation());
        assertEquals(2, strict.statements().size());
        assertEquals(2, relaxed.statements().size());
    }

    @Test
    public void sharedReadOnlyInputsHaveNoWriteDependence() {
        Matrix shared = new Matrix(2, 2);
        AffineProgram program = lower(MatrixCompiler.compile(
            MatrixExpr.input("A", shared).add(MatrixExpr.input("B", shared))));
        AffineStatement add = onlyStatement(program);

        assertEquals(DependenceStatus.PROVEN_NONE,
            program.dependenceGraph().query(add, add, DependenceKind.RAW));
        assertTrue(program.dependenceGraph().dependences().isEmpty());
    }

    @Test
    public void m1PlannerReferenceBehaviorRemainsAvailable() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(1000, 10))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(10, 1000)))
            .matmul(MatrixExpr.symbolicInput("C", new MatrixShape(1000, 10)));

        ExecutionPlan strict = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);
        ExecutionPlan relaxed = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);

        assertEquals(20_000_000L, strict.scalarMultiplicationCost());
        assertEquals(200_000L, relaxed.scalarMultiplicationCost());
        assertTrue(strict.expression().contains("matmul(\n  matmul("));
        assertTrue(relaxed.expression().contains("input(A),\n  matmul("));
    }

    private static AffineProgram lower(MatrixExpr expression) {
        return MatrixAffineCompiler.lower(MatrixCompiler.compile(expression));
    }

    private static AffineProgram lower(ExecutionPlan plan) {
        return MatrixAffineCompiler.lower(plan);
    }

    private static AffineStatement onlyStatement(AffineProgram program) {
        assertEquals(1, program.statements().size());
        return program.statements().get(0);
    }

    private static LogicalBuffer buffer(AffineProgram program, String name) {
        return program.buffers().stream()
            .filter(candidate -> candidate.name().equals(name))
            .findFirst()
            .orElseThrow(() -> new AssertionError("No buffer named " + name));
    }

    private static LogicalBuffer temporary(AffineProgram program, int index) {
        return program.buffers().stream()
            .filter(LogicalBuffer::isTemporary)
            .skip(index)
            .findFirst()
            .orElseThrow(() -> new AssertionError("No temporary at index " + index));
    }

    private static LogicalBuffer externalBuffer(int id) {
        return new LogicalBuffer(
            id,
            "external" + id,
            BufferKind.EXTERNAL_INPUT,
            BufferOwnership.BORROWED,
            MemorySpace.HEAP,
            new MatrixShape(2, 2));
    }

    private static String bracketedIndices(AffineAccess access) {
        return "[" + access.index(0) + "," + access.index(1) + "]";
    }

    private static long countStatementsOfKind(AffineProgram program, StatementKind kind) {
        return program.statements().stream().filter(statement -> statement.kind() == kind).count();
    }
}
