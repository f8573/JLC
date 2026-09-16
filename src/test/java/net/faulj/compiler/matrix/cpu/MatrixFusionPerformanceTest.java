package net.faulj.compiler.matrix.cpu;

import static org.junit.Assert.assertEquals;

import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Locale;
import java.util.function.Supplier;
import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.matrix.Matrix;

/** Isolated, adequately warmed M4 fusion measurement; informational only. */
public class MatrixFusionPerformanceTest {
    private static volatile double blackhole;

    @Test
    public void warmedFusionSizeCurveAndAllocation() {
        System.out.println("M4 warmed fusion methodology: deterministic inputs, "
            + "correctness before timing, each path independently warmed "
            + "for 5000/2000/1000/500/150 executions by ascending size, "
            + "31 measured, median, "
            + "execution only, allocation per thread over 100 executions");
        for (int size : new int[] {64, 128, 256, 512, 1024}) {
            Matrix a = filled(size, 0.001);
            Matrix b = filled(size, 0.002);
            CompiledMatrixProgram program = MatrixCompiler.compileProgram(
                MatrixExpr.input(a).scale(2.5).add(MatrixExpr.input(b)));
            Matrix reference = a.multiplyScalar(2.5).add(b);
            Matrix actual = program.execute();
            for (int p = 0; p < reference.getRawData().length; p++) {
                assertEquals(reference.getRawData()[p], actual.getRawData()[p], 0.0);
            }
            Supplier<Matrix> eager = () -> a.multiplyScalar(2.5).add(b);
            Supplier<Matrix> fused = program::execute;
            int warmups = switch (size) {
                case 64 -> 5000;
                case 128 -> 2000;
                case 256 -> 1000;
                case 512 -> 500;
                default -> 150;
            };
            double eagerMillis = medianMillis(eager, warmups);
            double fusedMillis = medianMillis(fused, warmups);
            long eagerAllocation = allocatedBytes(eager);
            long fusedAllocation = allocatedBytes(fused);
            System.out.printf(Locale.ROOT,
                "M4 FUSION CURVE size=%d eagerMedianMs=%.6f fusedMedianMs=%.6f "
                    + "eagerOverFused=%.3fx eagerAllocatedBytes=%d fusedAllocatedBytes=%d "
                    + "eagerPayloadBytes=%d fusedPayloadBytes=%d%n",
                size, eagerMillis, fusedMillis, eagerMillis / fusedMillis,
                eagerAllocation, fusedAllocation,
                (long) size * size * 16, (long) size * size * 8);
        }
    }

    private static double medianMillis(Supplier<Matrix> operation, int warmups) {
        for (int i = 0; i < warmups; i++) {
            blackhole = checksum(operation.get());
        }
        long[] elapsed = new long[31];
        for (int i = 0; i < elapsed.length; i++) {
            long start = System.nanoTime();
            Matrix result = operation.get();
            elapsed[i] = System.nanoTime() - start;
            blackhole = checksum(result);
        }
        Arrays.sort(elapsed);
        return elapsed[elapsed.length / 2] / 1_000_000.0;
    }

    private static long allocatedBytes(Supplier<Matrix> operation) {
        com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean)
            ManagementFactory.getThreadMXBean();
        if (!bean.isThreadAllocatedMemorySupported()) {
            return -1;
        }
        bean.setThreadAllocatedMemoryEnabled(true);
        long thread = Thread.currentThread().getId();
        long before = bean.getThreadAllocatedBytes(thread);
        for (int i = 0; i < 100; i++) {
            blackhole = checksum(operation.get());
        }
        return (bean.getThreadAllocatedBytes(thread) - before) / 100;
    }

    private static Matrix filled(int size, double step) {
        Matrix matrix = new Matrix(size, size);
        double[] data = matrix.getRawData();
        for (int p = 0; p < data.length; p++) {
            data[p] = step * (1 + (p * 17 % 101));
        }
        return matrix;
    }

    private static double checksum(Matrix matrix) {
        double sum = 0.0;
        for (double value : matrix.getRawData()) {
            sum += value;
        }
        return sum;
    }
}
