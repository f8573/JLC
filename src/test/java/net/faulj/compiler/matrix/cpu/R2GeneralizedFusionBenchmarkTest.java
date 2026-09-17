package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Locale;
import java.util.function.Supplier;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.matrix.Matrix;

/**
 * Opt-in R2 benchmark suite. Compilation is outside the timed section and
 * the default test run stays quiet; enable it with
 * {@code -Djlc.compiler.r2.benchmark=true}.
 */
public class R2GeneralizedFusionBenchmarkTest {
    private static volatile double blackhole;

    @Test
    public void benchmarkFusionModes() {
        if (!Boolean.getBoolean("jlc.compiler.r2.benchmark")) {
            return;
        }
        String previousMemory = System.getProperty(MemoryPlannerStrategy.PROPERTY);
        try {
            System.setProperty(
                MemoryPlannerStrategy.PROPERTY,
                System.getProperty("jlc.compiler.r2.benchmark.memory", "reuse"));
            for (int size : configuredSizes("jlc.compiler.r2.benchmark.sizes", 256, 512, 1024)) {
                runCase("A-long-chain", longChain(size), size);
                runCase("B-shared-dag", sharedDag(size), size);
                runCase("E-fusion-r1", fusionAndReuse(size), size);
                runCase("F-escaping-fanout", escapingFanout(size), size);
            }
            for (int[] shape : new int[][]{{512, 128}, {128, 512}, {512, 512}}) {
                runCase("C-transpose-pipeline", transposePipeline(shape[0], shape[1]),
                    shape[0] + "x" + shape[1]);
            }
            for (int size : configuredSizes("jlc.compiler.r2.gemm-sizes", 128, 192, 256)) {
                runCase("D-gemm-epilogue", gemmEpilogue(size), size);
            }
        } finally {
            if (previousMemory == null) {
                System.clearProperty(MemoryPlannerStrategy.PROPERTY);
            } else {
                System.setProperty(MemoryPlannerStrategy.PROPERTY, previousMemory);
            }
        }
    }

    private static void runCase(String name, BenchmarkCase benchmarkCase, Object size) {
        System.out.println("R2 benchmark workload=" + name + " size=" + size
            + " memory=" + System.getProperty(MemoryPlannerStrategy.PROPERTY));
        for (FusionStrategy strategy : FusionStrategy.values()) {
            CompiledMatrixProgram program = compile(benchmarkCase.expression(), strategy);
            assertMatrixEquals(benchmarkCase.expected(), program.execute());
            double medianMillis = medianMillis(program::execute, 2, 5);
            long allocationBytes = allocatedBytes(program::execute, 3);
            FusionMetrics metrics = program.cpuPlan().fusionMetrics();
            System.out.printf(Locale.ROOT,
                "R2_RESULT workload=%s size=%s mode=%s statements=%d "
                    + "elided=%d materialized=%d slots=%d passes=%d/%d "
                    + "readBytes=%d/%d writeBytes=%d/%d allocBytes=%d runtimeMs=%.6f "
                    + "path=%s%n",
                name, size, strategy.propertyValue(),
                program.affineProgram().statements().size(),
                metrics.fusionElidedTemporaryCount(),
                program.cpuPlan().materializedTemporaryCount(),
                program.cpuPlan().physicalSlotCount(),
                metrics.fullMatrixPassesBefore(), metrics.fullMatrixPassesAfter(),
                metrics.estimatedLogicalBytesReadBefore(), metrics.estimatedLogicalBytesReadAfter(),
                metrics.estimatedLogicalBytesWrittenBefore(), metrics.estimatedLogicalBytesWrittenAfter(),
                allocationBytes, medianMillis, executionPath(program));
        }
    }

    private static String executionPath(CompiledMatrixProgram program) {
        for (CpuStep step : program.cpuPlan().steps()) {
            if (step instanceof CpuFusedRegionStep fused) {
                return String.valueOf(fused.lastExecutionPath());
            }
            if (step instanceof CpuFusedElementwiseStep legacy) {
                return "LEGACY_" + legacy.lastExecutionPath();
            }
        }
        return "MATERIALIZED";
    }

    private static CompiledMatrixProgram compile(MatrixExpr expression,
                                                  FusionStrategy strategy) {
        return MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, strategy);
    }

    private static BenchmarkCase longChain(int size) {
        Matrix a = filled(size, size, 0.001);
        Matrix b = filled(size, size, 0.002);
        Matrix c = filled(size, size, 0.003);
        Matrix d = filled(size, size, 0.004);
        Matrix e = filled(size, size, 0.005);
        MatrixExpr expression = MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b))
            .scale(0.5).add(MatrixExpr.input(c)).scale(-1.25).transpose()
            .scale(0.75).add(MatrixExpr.input(d)).scale(1.1).add(MatrixExpr.input(e))
            .scale(0.9);
        return expected(expression);
    }

    private static BenchmarkCase sharedDag(int size) {
        Matrix a = filled(size, size, 0.001);
        Matrix b = filled(size, size, 0.002);
        MatrixExpr shared = MatrixExpr.input(a).scale(2.0);
        MatrixExpr expression = shared.add(MatrixExpr.input(b))
            .add(shared.scale(3.0)).scale(0.5);
        return expected(expression);
    }

    private static BenchmarkCase transposePipeline(int rows, int columns) {
        Matrix a = filled(rows, columns, 0.001);
        Matrix b = filled(columns, rows, 0.002);
        MatrixExpr expression = MatrixExpr.input(a).transpose().scale(1.5)
            .add(MatrixExpr.input(b)).scale(0.75);
        return expected(expression);
    }

    private static BenchmarkCase gemmEpilogue(int size) {
        Matrix a = filled(size, size, 0.001);
        Matrix b = filled(size, size, 0.002);
        Matrix c = filled(size, size, 0.003);
        Matrix d = filled(size, size, 0.004);
        MatrixExpr expression = MatrixExpr.input(a).matmul(MatrixExpr.input(b))
            .scale(2.0).add(MatrixExpr.input(c)).scale(0.5).add(MatrixExpr.input(d));
        return expected(expression);
    }

    private static BenchmarkCase fusionAndReuse(int size) {
        Matrix a = filled(size, size, 0.001);
        Matrix b = filled(size, size, 0.002);
        Matrix c = filled(size, size, 0.003);
        Matrix d = filled(size, size, 0.004);
        Matrix e = filled(size, size, 0.005);
        MatrixExpr first = MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)).scale(0.5);
        MatrixExpr expression = first.matmul(MatrixExpr.input(c))
            .add(MatrixExpr.input(d)).scale(0.75).add(MatrixExpr.input(e));
        return expected(expression);
    }

    private static BenchmarkCase escapingFanout(int size) {
        Matrix a = filled(size, size, 0.001);
        Matrix b = filled(size, size, 0.002);
        Matrix c = filled(size, size, 0.003);
        MatrixExpr shared = MatrixExpr.input(a).scale(2.0);
        MatrixExpr branch = shared.add(MatrixExpr.input(b));
        MatrixExpr expression = branch.add(shared.matmul(MatrixExpr.input(c)));
        return expected(expression);
    }

    private static BenchmarkCase expected(MatrixExpr expression) {
        return new BenchmarkCase(expression, MatrixCompiler.evaluate(expression));
    }

    private static double medianMillis(Supplier<Matrix> operation,
                                       int warmups,
                                       int measurements) {
        for (int index = 0; index < warmups; index++) {
            blackhole = checksum(operation.get());
        }
        long[] elapsed = new long[measurements];
        for (int index = 0; index < elapsed.length; index++) {
            long start = System.nanoTime();
            Matrix result = operation.get();
            elapsed[index] = System.nanoTime() - start;
            blackhole = checksum(result);
        }
        Arrays.sort(elapsed);
        return elapsed[elapsed.length / 2] / 1_000_000.0;
    }

    private static long allocatedBytes(Supplier<Matrix> operation, int iterations) {
        com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean)
            ManagementFactory.getThreadMXBean();
        if (!bean.isThreadAllocatedMemorySupported()) {
            return -1L;
        }
        bean.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().getId();
        long before = bean.getThreadAllocatedBytes(thread);
        for (int index = 0; index < iterations; index++) {
            blackhole = checksum(operation.get());
        }
        return (bean.getThreadAllocatedBytes(thread) - before) / iterations;
    }

    private static Matrix filled(int rows, int columns, double step) {
        Matrix matrix = new Matrix(rows, columns);
        double[] data = matrix.getRawData();
        for (int index = 0; index < data.length; index++) {
            data[index] = step * (1 + (index * 17L % 101));
        }
        return matrix;
    }

    private static int[] configuredSizes(String property, int... defaults) {
        String configured = System.getProperty(property);
        if (configured == null || configured.isBlank()) {
            return defaults;
        }
        return Arrays.stream(configured.split(","))
            .map(String::trim)
            .filter(value -> !value.isEmpty())
            .mapToInt(Integer::parseInt)
            .toArray();
    }

    private static double checksum(Matrix matrix) {
        double result = 0.0;
        for (double value : matrix.getRawData()) {
            result += value;
        }
        return result;
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

    private record BenchmarkCase(MatrixExpr expression, Matrix expected) {
    }
}
