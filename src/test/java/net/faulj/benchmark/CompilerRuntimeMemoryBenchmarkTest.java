package net.faulj.benchmark;

import static org.junit.Assert.assertEquals;

import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Locale;
import java.util.function.Supplier;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.CpuExecutionPlan;
import net.faulj.compiler.matrix.cpu.MemoryPlannerStrategy;
import net.faulj.compiler.matrix.cpu.PhysicalMemoryPlan;
import net.faulj.matrix.Matrix;

/**
 * Dedicated post-M4 R1 measurements. The benchmark is kept out of the
 * default correctness task; run it with {@code benchmarkTest}.
 */
public class CompilerRuntimeMemoryBenchmarkTest {
    private static final int WARMUPS = 1;
    private static final int MEASURED = 3;
    private static volatile double blackhole;

    @Test
    public void r1MemoryPlanningWorkloads() {
        System.out.println("R1 benchmark environment: java=" + System.getProperty("java.version")
            + ", os=" + System.getProperty("os.name")
            + ", arch=" + System.getProperty("os.arch")
            + ", processors=" + Runtime.getRuntime().availableProcessors()
            + ", warmups=" + WARMUPS + ", measured=" + MEASURED
            + ", statistic=median, allocation=ThreadMXBean average bytes/execution");

        for (int size : new int[] {256, 512, 1024}) {
            measure("A-long-elementwise-chain", size, elementwiseChain(size));
        }
        for (int size : new int[] {256, 512, 1024}) {
            measure("B-shared-forked-dag", size, sharedForkedDag(size));
        }
        for (int size : new int[] {128, 192, 256}) {
            measure("C-gemm-postprocessing", size, gemmPostprocessing(size));
        }
        for (int size : new int[] {128, 192, 256}) {
            measure("D-relaxed-matrix-chain", size, relaxedMatrixChain(size));
        }
        for (int size : new int[] {256, 512, 1024}) {
            measure("E-fusion-with-surrounding-ops", size, fusionWithSurroundingOps(size));
        }
    }

    private static void measure(String workload,
                                int size,
                                CompiledMatrixProgram program) {
        Matrix legacyResult = execute(program, MemoryPlannerStrategy.LEGACY);
        Matrix reuseResult = execute(program, MemoryPlannerStrategy.REUSE);
        assertMatrixEquals(legacyResult, reuseResult, 1e-8);

        PhysicalMemoryPlan memory = program.cpuPlan().physicalMemoryPlan();
        ModeMeasurement legacy = measureMode(program, MemoryPlannerStrategy.LEGACY);
        ModeMeasurement reuse = measureMode(program, MemoryPlannerStrategy.REUSE);
        String allocationReduction = percentage(legacy.allocatedBytes, reuse.allocatedBytes);
        double runtimeRatio = legacy.medianMillis / reuse.medianMillis;

        System.out.printf(Locale.ROOT,
            "R1 MEMORY workload=%s size=%d logicalTemps=%d materializedTemps=%d "
                + "fusionElided=%d physicalSlots=%d logicalBytes=%d peakLiveBytes=%d "
                + "physicalBytes=%d logicalToPhysicalReuse=%d legacyAllocations=%d "
                + "reuseAllocations=%d%n",
            workload, size, memory.logicalTemporaryCount(),
            memory.materializedLogicalTemporaryCount(), memory.elidedLogicalTemporaryCount(),
            memory.physicalSlotCount(), memory.logicalTemporaryBytesSum(),
            memory.peakLiveTemporaryBytes(), memory.physicalTemporaryBytes(),
            memory.logicalToPhysicalReuseCount(), memory.legacyAllocationCount(),
            memory.allocationCount());
        System.out.printf(Locale.ROOT,
            "R1 ALLOCATION workload=%s size=%d legacyAllocBytes=%d reuseAllocBytes=%d "
                + "reduction=%s%n",
            workload, size, legacy.allocatedBytes, reuse.allocatedBytes, allocationReduction);
        System.out.printf(Locale.ROOT,
            "R1 RUNTIME workload=%s size=%d legacyMedianMs=%.3f reuseMedianMs=%.3f "
                + "legacyOverReuse=%.3fx correctness=PASS%n",
            workload, size, legacy.medianMillis, reuse.medianMillis, runtimeRatio);
    }

    private static ModeMeasurement measureMode(CompiledMatrixProgram program,
                                               MemoryPlannerStrategy strategy) {
        String previous = System.getProperty(MemoryPlannerStrategy.PROPERTY);
        try {
            System.setProperty(MemoryPlannerStrategy.PROPERTY, strategy.propertyValue());
            for (int iteration = 0; iteration < WARMUPS; iteration++) {
                blackhole = checksum(program.execute());
            }
            long[] elapsed = new long[MEASURED];
            for (int iteration = 0; iteration < MEASURED; iteration++) {
                long start = System.nanoTime();
                blackhole = checksum(program.execute());
                elapsed[iteration] = System.nanoTime() - start;
            }
            Arrays.sort(elapsed);
            double medianMillis = elapsed[elapsed.length / 2] / 1_000_000.0;
            long allocatedBytes = allocatedBytes(program::execute);
            return new ModeMeasurement(medianMillis, allocatedBytes);
        } finally {
            restoreProperty(previous);
        }
    }

    private static long allocatedBytes(Supplier<Matrix> operation) {
        com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean)
            ManagementFactory.getThreadMXBean();
        if (!bean.isThreadAllocatedMemorySupported()) {
            return -1L;
        }
        bean.setThreadAllocatedMemoryEnabled(true);
        long threadId = Thread.currentThread().getId();
        for (int iteration = 0; iteration < WARMUPS; iteration++) {
            blackhole = checksum(operation.get());
        }
        long before = bean.getThreadAllocatedBytes(threadId);
        for (int iteration = 0; iteration < MEASURED; iteration++) {
            blackhole = checksum(operation.get());
        }
        return (bean.getThreadAllocatedBytes(threadId) - before) / MEASURED;
    }

    private static Matrix execute(CompiledMatrixProgram program,
                                  MemoryPlannerStrategy strategy) {
        String previous = System.getProperty(MemoryPlannerStrategy.PROPERTY);
        try {
            System.setProperty(MemoryPlannerStrategy.PROPERTY, strategy.propertyValue());
            return program.execute();
        } finally {
            restoreProperty(previous);
        }
    }

    private static CompiledMatrixProgram elementwiseChain(int size) {
        MatrixExpr expression = MatrixExpr.input("A", filled(size, size, 0.0001));
        for (int round = 0; round < 4; round++) {
            expression = expression.scale(1.0001 + round * 0.0001)
                .add(MatrixExpr.input("B" + round, filled(size, size, 0.0002 + round * 0.00001)))
                .transpose();
        }
        return MatrixCompiler.compileProgram(expression, OptimizationSemantics.STRICT);
    }

    private static CompiledMatrixProgram sharedForkedDag(int size) {
        MatrixExpr shared = MatrixExpr.input("A", filled(size, size, 0.0001))
            .add(MatrixExpr.input("B", filled(size, size, 0.0002)));
        MatrixExpr left = shared.scale(1.5);
        MatrixExpr right = shared.transpose().scale(0.5);
        return MatrixCompiler.compileProgram(left.add(right), OptimizationSemantics.STRICT);
    }

    private static CompiledMatrixProgram gemmPostprocessing(int size) {
        MatrixExpr product = MatrixExpr.input("A", filled(size, size, 0.00001))
            .matmul(MatrixExpr.input("B", filled(size, size, 0.00002)));
        MatrixExpr expression = product.scale(1.01)
            .add(MatrixExpr.input("C", filled(size, size, 0.00003)))
            .transpose()
            .scale(0.99);
        return MatrixCompiler.compileProgram(expression, OptimizationSemantics.STRICT);
    }

    private static CompiledMatrixProgram relaxedMatrixChain(int size) {
        int inner = Math.max(16, size / 4);
        MatrixExpr expression = MatrixExpr.input("A", filled(size, inner, 0.00001))
            .matmul(MatrixExpr.input("B", filled(inner, size, 0.00002)))
            .matmul(MatrixExpr.input("C", filled(size, inner, 0.00003)));
        return MatrixCompiler.compileProgram(expression, OptimizationSemantics.RELAXED);
    }

    private static CompiledMatrixProgram fusionWithSurroundingOps(int size) {
        MatrixExpr expression = MatrixExpr.input("A", filled(size, size, 0.0001))
            .scale(2.5)
            .add(MatrixExpr.input("B", filled(size, size, 0.0002)))
            .transpose()
            .scale(0.5)
            .add(MatrixExpr.input("C", filled(size, size, 0.0003)));
        return MatrixCompiler.compileProgram(expression, OptimizationSemantics.STRICT);
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

    private static double checksum(Matrix matrix) {
        double result = 0.0;
        for (int row = 0; row < matrix.getRowCount(); row++) {
            for (int column = 0; column < matrix.getColumnCount(); column++) {
                result += matrix.get(row, column);
            }
        }
        return result;
    }

    private static void assertMatrixEquals(Matrix expected, Matrix actual, double delta) {
        assertEquals(expected.getRowCount(), actual.getRowCount());
        assertEquals(expected.getColumnCount(), actual.getColumnCount());
        for (int row = 0; row < expected.getRowCount(); row++) {
            for (int column = 0; column < expected.getColumnCount(); column++) {
                assertEquals("at [" + row + "," + column + "]",
                    expected.get(row, column), actual.get(row, column), delta);
            }
        }
    }

    private static String percentage(long legacy, long reuse) {
        if (legacy < 0 || reuse < 0 || legacy == 0) {
            return "unavailable";
        }
        return String.format(Locale.ROOT, "%.1f%%", 100.0 * (legacy - reuse) / legacy);
    }

    private static void restoreProperty(String previous) {
        if (previous == null) {
            System.clearProperty(MemoryPlannerStrategy.PROPERTY);
        } else {
            System.setProperty(MemoryPlannerStrategy.PROPERTY, previous);
        }
    }

    private record ModeMeasurement(double medianMillis, long allocatedBytes) {
    }
}
