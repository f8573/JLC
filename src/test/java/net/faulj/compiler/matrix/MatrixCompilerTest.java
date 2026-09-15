package net.faulj.compiler.matrix;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.Test;

import net.faulj.matrix.Matrix;
import net.faulj.matrix.OffHeapMatrix;

/**
 * Focused M1 coverage for the matrix expression IR, planner, and interpreter.
 */
public class MatrixCompilerTest {
    @Test
    public void infersShapesForCoreNodes() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(3, 4));
        MatrixExpr product = a.matmul(b);

        assertEquals(new MatrixShape(2, 4), product.shape());
        assertEquals(new MatrixShape(2, 4), product.add(MatrixExpr.input(new Matrix(2, 4))).shape());
        assertEquals(new MatrixShape(4, 2), product.transpose().shape());
        assertEquals(new MatrixShape(2, 4), product.scale(2.0).shape());
    }

    @Test
    public void rejectsInvalidMatMulShapeImmediately() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(4, 5));

        try {
            a.matmul(b);
            fail("Expected incompatible MatMul shapes to be rejected");
        } catch (IllegalArgumentException exception) {
            assertTrue(exception.getMessage().contains("2x3"));
            assertTrue(exception.getMessage().contains("4x5"));
            assertTrue(exception.getMessage().contains("left columns"));
        }
    }

    @Test
    public void rejectsInvalidAddShapeImmediately() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(2, 4));

        try {
            a.add(b);
            fail("Expected incompatible Add shapes to be rejected");
        } catch (IllegalArgumentException exception) {
            assertTrue(exception.getMessage().contains("2x3"));
            assertTrue(exception.getMessage().contains("2x4"));
            assertTrue(exception.getMessage().contains("identical"));
        }
    }

    @Test
    public void safeCanonicalizationPreservesIdentity() {
        MatrixExpr input = MatrixExpr.input("A", new Matrix(2, 3));

        assertSame(input, input.transpose().transpose());
        assertSame(input, input.scale(1.0));

        MatrixExpr directPair = new Transpose(new Transpose(input));
        assertEquals(MatrixCompiler.compile(directPair).expression(), "input(A)");
    }

    @Test
    public void sharedProducerRemainsOnePlanNode() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(3, 2));
        MatrixExpr shared = a.matmul(b);

        ExecutionPlan plan = MatrixCompiler.compile(shared.add(shared));

        assertTrue(plan.root() instanceof PlanAdd);
        PlanAdd add = (PlanAdd) plan.root();
        assertSame(add.lhs(), add.rhs());
        assertEquals(1, countLinesContaining(plan.dump(), "= matmul "));
        assertTrue(plan.dump().contains("add %2, %2"));
    }

    @Test
    public void relaxedChainOptimizationDoesNotDuplicateSharedProducer() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(3, 2));
        MatrixExpr c = MatrixExpr.input("C", new Matrix(2, 2));
        MatrixExpr shared = a.matmul(b);
        MatrixExpr expression = shared.matmul(c).add(shared);

        ExecutionPlan plan = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);

        PlanAdd add = (PlanAdd) plan.root();
        PlanMatMul chain = (PlanMatMul) add.lhs();
        assertSame(chain.lhs(), add.rhs());
        assertEquals(2, countLinesContaining(plan.dump(), "= matmul "));
    }

    @Test
    public void planRenderingIsDeterministic() {
        MatrixExpr a = MatrixExpr.input("A", new Matrix(2, 3));
        MatrixExpr b = MatrixExpr.input("B", new Matrix(3, 4));
        MatrixExpr c = MatrixExpr.input("C", new Matrix(4, 2));
        MatrixExpr expression = a.matmul(b).matmul(c).transpose();

        ExecutionPlan first = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);
        ExecutionPlan second = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);

        assertEquals(first.expression(), second.expression());
        assertEquals(first.dump(), second.dump());
        assertFalse(first.dump().isEmpty());
    }

    @Test
    public void strictSemanticsPreservesMatMulAssociation() {
        MatrixExpr a = MatrixExpr.symbolicInput("A", new MatrixShape(1000, 10));
        MatrixExpr b = MatrixExpr.symbolicInput("B", new MatrixShape(10, 1000));
        MatrixExpr c = MatrixExpr.symbolicInput("C", new MatrixShape(1000, 10));
        MatrixExpr expression = a.matmul(b).matmul(c);

        ExecutionPlan plan = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);

        assertTrue(plan.root() instanceof PlanMatMul);
        PlanMatMul root = (PlanMatMul) plan.root();
        assertTrue(root.lhs() instanceof PlanMatMul);
        assertTrue(root.rhs() instanceof PlanInput);
        assertEquals(20_000_000L, plan.scalarMultiplicationCost());
        assertTrue(plan.expression().startsWith("matmul(\n  matmul("));
    }

    @Test
    public void relaxedSemanticsChoosesMinimumCostReferenceAssociation() {
        MatrixExpr a = MatrixExpr.symbolicInput("A", new MatrixShape(1000, 10));
        MatrixExpr b = MatrixExpr.symbolicInput("B", new MatrixShape(10, 1000));
        MatrixExpr c = MatrixExpr.symbolicInput("C", new MatrixShape(1000, 10));
        MatrixExpr expression = a.matmul(b).matmul(c);

        ExecutionPlan strict = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);
        ExecutionPlan relaxed = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);

        assertEquals(20_000_000L, strict.scalarMultiplicationCost());
        assertEquals(200_000L, relaxed.scalarMultiplicationCost());
        assertEquals(100L, strict.scalarMultiplicationCost() / relaxed.scalarMultiplicationCost());

        assertTrue(relaxed.root() instanceof PlanMatMul);
        PlanMatMul root = (PlanMatMul) relaxed.root();
        assertTrue(root.lhs() instanceof PlanInput);
        assertTrue(root.rhs() instanceof PlanMatMul);
        assertEquals("matmul(\n  input(A),\n  matmul(\n    input(B),\n    input(C)\n  )\n)", relaxed.expression());
        assertEquals(8_080_000L, strict.estimatedOutputBytes());
        assertEquals(80_800L, relaxed.estimatedOutputBytes());
    }

    @Test
    public void fastSemanticsUsesOnlyCurrentRelaxedPermission() {
        MatrixExpr a = MatrixExpr.symbolicInput("A", new MatrixShape(1000, 10));
        MatrixExpr b = MatrixExpr.symbolicInput("B", new MatrixShape(10, 1000));
        MatrixExpr c = MatrixExpr.symbolicInput("C", new MatrixShape(1000, 10));

        ExecutionPlan fast = MatrixCompiler.compile(a.matmul(b).matmul(c), OptimizationSemantics.FAST);

        assertEquals(200_000L, fast.scalarMultiplicationCost());
        assertEquals(OptimizationSemantics.FAST, fast.semantics());
    }

    @Test
    public void relaxedAndStrictPlansAreNumericallyEquivalent() {
        Matrix a = new Matrix(new double[][]{{1, -2}, {3, 4}, {-5, 6}, {7, 8}});
        Matrix b = new Matrix(new double[][]{{2, 1, -1, 3, 4}, {0, -2, 5, 1, -3}});
        Matrix c = new Matrix(new double[][]{{1, 2, 0}, {-1, 3, 4}, {2, -2, 1}, {0, 1, -3}, {4, 0, 2}});
        MatrixExpr expression = MatrixExpr.input("A", a)
            .matmul(MatrixExpr.input("B", b))
            .matmul(MatrixExpr.input("C", c));

        Matrix strict = MatrixCompiler.evaluate(expression, OptimizationSemantics.STRICT);
        Matrix relaxed = MatrixCompiler.evaluate(expression, OptimizationSemantics.RELAXED);

        assertMatrixEquals(strict, relaxed, 1.0e-10);
        assertMatrixEquals(a.multiply(b).multiply(c), relaxed, 1.0e-10);
    }

    @Test
    public void randomizedCompatibleChainsRemainEquivalent() {
        Random random = new Random(0x4d41545249584d31L);
        for (int trial = 0; trial < 20; trial++) {
            int[] dimensions = new int[6];
            for (int i = 0; i < dimensions.length; i++) {
                dimensions[i] = 1 + random.nextInt(5);
            }

            Matrix[] matrices = new Matrix[5];
            MatrixExpr expression = null;
            for (int i = 0; i < matrices.length; i++) {
                matrices[i] = new Matrix(dimensions[i], dimensions[i + 1]);
                fill(matrices[i], random);
                MatrixExpr input = MatrixExpr.input("M" + i, matrices[i]);
                expression = expression == null ? input : expression.matmul(input);
            }

            ExecutionPlan strictPlan = MatrixCompiler.compile(expression, OptimizationSemantics.STRICT);
            ExecutionPlan relaxedPlan = MatrixCompiler.compile(expression, OptimizationSemantics.RELAXED);
            Matrix strict = strictPlan.evaluate();
            Matrix relaxed = relaxedPlan.evaluate();

            assertTrue(relaxedPlan.scalarMultiplicationCost() <= strictPlan.scalarMultiplicationCost());
            assertMatrixEquals(strict, relaxed, 1.0e-8);
        }
    }

    @Test
    public void evaluationDoesNotModifyHeapInputs() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        double[] originalA = a.getRawData().clone();
        double[] originalB = b.getRawData().clone();
        MatrixExpr expression = MatrixExpr.input("A", a)
            .matmul(MatrixExpr.input("B", b))
            .add(MatrixExpr.input("A2", new Matrix(new double[][]{{1, 1}, {1, 1}})))
            .scale(0.5)
            .transpose();

        MatrixCompiler.evaluate(expression);

        assertArrayEquals(originalA, a.getRawData(), 0.0);
        assertArrayEquals(originalB, b.getRawData(), 0.0);
    }

    @Test
    public void evaluatesHeapMatrixWithExistingOperations() {
        Matrix a = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});
        MatrixExpr expression = MatrixExpr.input(a)
            .matmul(MatrixExpr.input(b))
            .add(MatrixExpr.input(b))
            .scale(0.5)
            .transpose();

        Matrix actual = MatrixCompiler.evaluate(expression);
        Matrix expected = a.multiply(b).add(b).multiplyScalar(0.5).transpose();

        assertMatrixEquals(expected, actual, 1.0e-12);
    }

    @Test
    public void evaluatesSupportedOffHeapInputThroughGemmFacade() {
        try (OffHeapMatrix a = new OffHeapMatrix(2, 2)) {
            a.set(0, 0, 1.0);
            a.set(0, 1, 2.0);
            a.set(1, 0, 3.0);
            a.set(1, 1, 4.0);
            a.syncToOffHeap();
            Matrix b = new Matrix(new double[][]{{5, 6}, {7, 8}});

            Matrix result = MatrixCompiler.evaluate(
                MatrixExpr.input("offHeapA", a).matmul(MatrixExpr.input("B", b)));
            try {
                assertEquals(19.0, result.get(0, 0), 1.0e-12);
                assertEquals(22.0, result.get(0, 1), 1.0e-12);
                assertEquals(43.0, result.get(1, 0), 1.0e-12);
                assertEquals(50.0, result.get(1, 1), 1.0e-12);
                assertTrue(result instanceof OffHeapMatrix);
            } finally {
                if (result instanceof OffHeapMatrix offHeapResult) {
                    offHeapResult.close();
                }
            }
        }
    }

    @Test
    public void symbolicInputsAreNotExecutable() {
        MatrixExpr expression = MatrixExpr.symbolicInput("A", new MatrixShape(2, 3))
            .matmul(MatrixExpr.symbolicInput("B", new MatrixShape(3, 2)));

        try {
            MatrixCompiler.evaluate(expression);
            fail("Expected symbolic input evaluation to fail");
        } catch (IllegalStateException exception) {
            assertTrue(exception.getMessage().contains("symbolic input"));
        }
    }

    @Test
    public void costArithmeticUsesLongAndSaturatesOverflow() {
        MatrixShape huge = new MatrixShape(Integer.MAX_VALUE, Integer.MAX_VALUE);
        CostEstimate cost = new FlopCostModel().estimateMatMul(huge, huge);

        assertEquals(Long.MAX_VALUE, cost.scalarMultiplications());
        assertEquals(Long.MAX_VALUE, cost.estimatedOutputBytes());
    }

    private static void fill(Matrix matrix, Random random) {
        for (int row = 0; row < matrix.getRowCount(); row++) {
            for (int col = 0; col < matrix.getColumnCount(); col++) {
                matrix.set(row, col, random.nextDouble() - 0.5);
            }
        }
    }

    private static int countLinesContaining(String text, String needle) {
        int count = 0;
        for (String line : text.split("\\n")) {
            if (line.contains(needle)) {
                count++;
            }
        }
        return count;
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual, double tolerance) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int row = 0; row < expected.getRowCount(); row++) {
            for (int col = 0; col < expected.getColumnCount(); col++) {
                assertEquals("Mismatch at (" + row + "," + col + ")",
                    expected.get(row, col), actual.get(row, col), tolerance);
            }
        }
    }
}
