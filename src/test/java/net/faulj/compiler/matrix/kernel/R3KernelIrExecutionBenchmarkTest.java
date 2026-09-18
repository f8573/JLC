package net.faulj.compiler.matrix.kernel;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.IdentityHashMap;
import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.CpuBufferBinding;
import net.faulj.compiler.matrix.cpu.CpuFusedRegionStep;
import net.faulj.compiler.matrix.cpu.FusedRegionPlan;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.matrix.Matrix;

/** Opt-in informational comparison of R2's scalar fast path and R3 reference execution. */
public class R3KernelIrExecutionBenchmarkTest {
    @Test
    public void executionComparison() {
        if (!Boolean.getBoolean("jlc.compiler.r3.executionBenchmark")) {
            return;
        }
        String previousKernelMode = System.getProperty(KernelIrMode.PROPERTY);
        try {
            System.setProperty(KernelIrMode.PROPERTY, "off");
            List<Workload> workloads = List.of(
                new Workload("scale-add 37x11", generalized(
                    MatrixExpr.input(matrix(37, 11, 1.0))
                        .scale(2.0).add(MatrixExpr.input(matrix(37, 11, 3.0))))),
                new Workload("transpose 37x11", generalized(
                    MatrixExpr.input(matrix(37, 11, 1.0)).transpose()
                        .scale(2.0).add(MatrixExpr.input(matrix(11, 37, 3.0))))),
                new Workload("shared producer 64x64", generalized(sharedExpression())));

            System.out.println("R3 Kernel IR reference execution comparison");
            System.out.println("| Workload | R2 fast path us | R3 scalar reference us | Ratio | Correct |");
            System.out.println("| -------- | ---------------: | ---------------------: | ----: | :------ |");
            for (Workload workload : workloads) {
                FusedRegionPlan region = workload.program.cpuPlan().fusedRegions().get(0);
                KernelLoweringResult lowering = KernelLowerer.lower(region);
                assertTrue(lowering.isEligible());
                long[] r2 = new long[5];
                long[] r3 = new long[5];
                Matrix expected = null;
                Matrix actual = null;
                for (int warmup = 0; warmup < 2; warmup++) {
                    expected = workload.program.execute();
                    actual = executeLowered(workload.program, lowering);
                }
                for (int sample = 0; sample < r2.length; sample++) {
                    long start = System.nanoTime();
                    expected = workload.program.execute();
                    r2[sample] = System.nanoTime() - start;
                    start = System.nanoTime();
                    actual = executeLowered(workload.program, lowering);
                    r3[sample] = System.nanoTime() - start;
                }
                assertArrayEquals(expected.getRawData(), actual.getRawData(), 0.0);
                CpuFusedRegionStep step = (CpuFusedRegionStep) workload.program.cpuPlan().steps().get(0);
                assertEquals(CpuFusedRegionStep.ExecutionPath.FAST_REAL_DENSE,
                    step.lastExecutionPath());
                double r2Micros = median(r2) / 1_000.0;
                double r3Micros = median(r3) / 1_000.0;
                double ratio = r2Micros == 0.0 ? Double.NaN : r3Micros / r2Micros;
                System.out.printf("| %s | %.1f | %.1f | %.2f | yes |%n",
                    workload.name, r2Micros, r3Micros, ratio);
            }
        } finally {
            if (previousKernelMode == null) {
                System.clearProperty(KernelIrMode.PROPERTY);
            } else {
                System.setProperty(KernelIrMode.PROPERTY, previousKernelMode);
            }
        }
    }

    private static CompiledMatrixProgram generalized(MatrixExpr expression) {
        return MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
    }

    private static MatrixExpr sharedExpression() {
        Matrix a = matrix(64, 64, 1.0);
        MatrixExpr shared = MatrixExpr.input(a).scale(3.0);
        return shared.add(shared).add(shared);
    }

    private static Matrix executeLowered(CompiledMatrixProgram program,
                                         KernelLoweringResult lowering) {
        FusedRegionPlan region = program.cpuPlan().fusedRegions().get(0);
        IdentityHashMap<LogicalBuffer, Matrix> logical = new IdentityHashMap<>();
        for (CpuBufferBinding binding : program.cpuPlan().inputBindings()) {
            logical.put(binding.buffer(), binding.matrix());
        }
        logical.put(region.outputBuffer(), new Matrix(
            region.outputBuffer().shape().rows(), region.outputBuffer().shape().columns()));
        KernelBinding binding = KernelBinding.fromLogicalBuffers(
            lowering.program().function(), logical);
        return KernelReferenceExecutor.executeVerified(lowering.program(), binding);
    }

    private static Matrix matrix(int rows, int columns, double value) {
        double[] data = new double[rows * columns];
        java.util.Arrays.fill(data, value);
        return Matrix.wrap(data, rows, columns);
    }

    private static long median(long[] values) {
        java.util.Arrays.sort(values);
        return values[values.length / 2];
    }

    private record Workload(String name, CompiledMatrixProgram program) {
    }
}
