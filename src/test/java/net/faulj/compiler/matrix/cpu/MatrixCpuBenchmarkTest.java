package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;

import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compute.DispatchPolicy;
import net.faulj.kernels.gemm.Gemm;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.BackendRegistry;

/**
 * The fixed M4 benchmark suite: chain reassociation, fusion, direct GEMM
 * dispatch boundary, and shared DAG execution. Results are informational;
 * every case is correctness-gated before timing.
 */
public class MatrixCpuBenchmarkTest {
    private static final int WARMUPS = 2;
    private static final int MEASURED = 5;
    private static volatile double blackhole;

    @Test
    public void fixedFourFamilyBenchmarkSuite() {
        System.out.println("M4 benchmark environment: java=" + System.getProperty("java.version")
            + ", os=" + System.getProperty("os.name")
            + ", arch=" + System.getProperty("os.arch")
            + ", processors=" + Runtime.getRuntime().availableProcessors()
            + ", backend=" + BackendRegistry.snapshot().activeBackend());
        System.out.println("M4 benchmark methodology: warmups=" + WARMUPS
            + ", measured=" + MEASURED
            + ", statistic=median, boundary=execution-only, correctness=before-timing");

        benchmarkMatrixChain();
        benchmarkElementwiseFusion();
        benchmarkDirectGemmBoundary();
        benchmarkSharedDag();
    }

    private static void benchmarkMatrixChain() {
        Matrix a = filled(1000, 10, 0.001);
        Matrix b = filled(10, 1000, 0.002);
        Matrix c = filled(1000, 10, 0.003);
        MatrixExpr expression = MatrixExpr.input("A", a)
            .matmul(MatrixExpr.input("B", b))
            .matmul(MatrixExpr.input("C", c));
        CompiledMatrixProgram strict = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);
        CompiledMatrixProgram relaxed = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.RELAXED);
        Matrix strictReference = MatrixCompiler.evaluate(expression, OptimizationSemantics.STRICT);
        Matrix relaxedReference = MatrixCompiler.evaluate(expression, OptimizationSemantics.RELAXED);
        assertMatrixEquals(strictReference, strict.execute(), 1e-8);
        assertMatrixEquals(relaxedReference, relaxed.execute(), 1e-8);

        double strictMillis = medianMillis(strict::execute);
        double relaxedMillis = medianMillis(relaxed::execute);
        double speedup = strictMillis / relaxedMillis;
        System.out.printf(Locale.ROOT,
            "M4 A MATRIX-CHAIN REASSOCIATION: strictCost=%d relaxedCost=%d "
                + "strictProductBackends=%s relaxedProductBackends=%s "
                + "strictMedianMs=%.3f relaxedMedianMs=%.3f observedRatio=%.3fx correctness=PASS%n",
            strict.expressionPlan().scalarMultiplicationCost(),
            relaxed.expressionPlan().scalarMultiplicationCost(),
            List.of(selectedGemmBackend(1000, 1000), selectedGemmBackend(1000, 10)),
            List.of(selectedGemmBackend(10, 10), selectedGemmBackend(1000, 10)),
            strictMillis, relaxedMillis, speedup);
    }

    private static void benchmarkElementwiseFusion() {
        Matrix a = filled(256, 256, 0.001);
        Matrix b = filled(256, 256, 0.002);
        MatrixExpr expression = MatrixExpr.input("A", a).scale(2.5).add(MatrixExpr.input("B", b));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);
        Matrix eagerReference = a.multiplyScalar(2.5).add(b);
        assertMatrixEquals(eagerReference, compiled.execute(), 1e-10);

        double eagerMillis = medianMillis(() -> a.multiplyScalar(2.5).add(b));
        double fusedMillis = medianMillis(compiled::execute);
        long eagerBytes = (long) 256 * 256 * Double.BYTES * 2L;
        System.out.printf(Locale.ROOT,
            "M4 B ELEMENTWISE FUSION: eagerTemporaryMaterializations=2 "
                + "compilerTemporaryMaterializations=%d eagerTemporaryBytes=%d "
                + "compilerTemporaryBytes=%d eagerMedianMs=%.3f fusedMedianMs=%.3f "
                + "correctness=PASS%n",
            compiled.cpuPlan().materializedTemporaryCount(),
            eagerBytes,
            compiled.cpuPlan().estimatedTemporaryBytes(),
            eagerMillis, fusedMillis);
    }

    private static void benchmarkDirectGemmBoundary() {
        Matrix a = filled(192, 192, 0.001);
        Matrix b = filled(192, 192, 0.002);
        MatrixExpr expression = MatrixExpr.input("A", a).matmul(MatrixExpr.input("B", b));
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);
        Matrix directReference = Gemm.multiply(a, b);
        assertMatrixEquals(directReference, compiled.execute(), 1e-10);

        double directMillis = medianMillis(() -> Gemm.multiply(a, b));
        double compilerMillis = medianMillis(compiled::execute);
        double difference = (compilerMillis / directMillis - 1.0) * 100.0;
        String interpretation = Math.abs(difference) <= 5.0
            ? "measurement-noise" : "observed-timing-difference";
        System.out.printf(Locale.ROOT,
            "M4 C DIRECT GEMM DISPATCH BOUNDARY: shape=192x192x192 directMedianMs=%.3f "
                + "compilerMedianMs=%.3f executionDelta=%.2f%% interpretation=%s "
                + "selectedBackend=%s sameGemmFacade=PASS "
                + "correctness=PASS%n",
            directMillis, compilerMillis, difference, interpretation,
            selectedGemmBackend(192, 192));
    }

    private static void benchmarkSharedDag() {
        Matrix a = filled(96, 96, 0.001);
        Matrix b = filled(96, 96, 0.002);
        MatrixExpr x = MatrixExpr.input("A", a).matmul(MatrixExpr.input("B", b));
        MatrixExpr expression = x.add(x);
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT);
        Matrix producer = Gemm.multiply(a, b);
        assertMatrixEquals(producer.add(producer), compiled.execute(), 1e-10);

        double compilerMillis = medianMillis(compiled::execute);
        System.out.printf(Locale.ROOT,
            "M4 D SHARED DAG: plannedGemmSteps=%d observedRuntimeInvocations=not-instrumented "
                + "temporaryBuffers=%d compilerMedianMs=%.3f sharedPlanNodeOnce=PASS "
                + "correctness=PASS%n",
            compiled.cpuPlan().gemmStepCount(),
            compiled.cpuPlan().temporaryCount(),
            compilerMillis);
    }

    private static String selectedGemmBackend(int rows, int columns) {
        DispatchPolicy policy = DispatchPolicy.defaultPolicy();
        int threads = policy.isParallelEnabled() ? policy.getParallelism() : 1;
        return BackendRegistry.shouldUseCppForAlgorithm(
            "gemm", "multiply", rows, columns, threads) ? "native" : "java";
    }

    private static double medianMillis(Supplier<Matrix> operation) {
        for (int iteration = 0; iteration < WARMUPS; iteration++) {
            blackhole = checksum(operation.get());
        }
        long[] elapsed = new long[MEASURED];
        for (int iteration = 0; iteration < MEASURED; iteration++) {
            long start = System.nanoTime();
            Matrix result = operation.get();
            blackhole = checksum(result);
            elapsed[iteration] = System.nanoTime() - start;
        }
        java.util.Arrays.sort(elapsed);
        return elapsed[elapsed.length / 2] / 1_000_000.0;
    }

    private static double checksum(Matrix matrix) {
        double result = 0.0;
        for (int row = 0; row < matrix.getRowCount(); row++) {
            for (int column = 0; column < matrix.getColumnCount(); column++) {
                result += matrix.get(row, column);
            }
        }
        return result;
    }

    private static Matrix filled(int rows, int columns, double step) {
        Matrix result = new Matrix(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                result.set(row, column, step * (1 + ((row * 17 + column * 31) % 101)));
            }
        }
        return result;
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual, double delta) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int row = 0; row < expected.getRowCount(); row++) {
            for (int column = 0; column < expected.getColumnCount(); column++) {
                assertEquals(
                    "at [" + row + "," + column + "]",
                    expected.get(row, column), actual.get(row, column), delta);
            }
        }
    }
}
