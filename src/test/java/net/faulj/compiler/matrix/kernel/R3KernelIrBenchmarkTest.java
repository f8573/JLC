package net.faulj.compiler.matrix.kernel;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.FusedRegionPlan;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.matrix.Matrix;

/** Opt-in compiler-only scaling audit for the R2-to-R3 lowering. */
public class R3KernelIrBenchmarkTest {
    @Test
    public void loweringAndVerificationScalingAudit() {
        if (!Boolean.getBoolean("jlc.compiler.r3.benchmark")) {
            return;
        }
        List<Row> rows = new ArrayList<>();
        for (int operationCount : new int[]{2, 10, 25, 50, 100}) {
            FusedRegionPlan region = regionWithScaleChain(operationCount);
            for (int warmup = 0; warmup < 3; warmup++) {
                KernelLowerer.lower(region);
            }
            long[] lowering = new long[7];
            long[] verification = new long[7];
            KernelLoweringResult last = null;
            for (int sample = 0; sample < lowering.length; sample++) {
                last = KernelLowerer.lower(region);
                lowering[sample] = last.loweringTimeNanos();
                verification[sample] = last.verificationTimeNanos();
            }
            assertTrue(last.isEligible());
            int expectedValues = 1 + operationCount * 2;
            int expectedOperations = expectedValues + 1;
            assertEquals(expectedValues, last.program().function().values().size());
            assertEquals(expectedOperations,
                last.program().function().body().operations().size());
            rows.add(new Row(operationCount, medianMicros(lowering),
                medianMicros(verification), expectedValues, expectedOperations));
        }

        assertTrue(rows.get(rows.size() - 1).irValues
            < rows.get(0).irValues * 50);
        System.out.println("R3 Kernel IR lowering/verification benchmark");
        System.out.println("| Ops | Lowering us | Verification us | IR values | IR operations |");
        System.out.println("| --: | -----------: | ---------------: | --------: | -------------: |");
        for (Row row : rows) {
            System.out.printf("| %d | %d | %d | %d | %d |%n",
                row.operations, row.loweringMicros, row.verificationMicros,
                row.irValues, row.irOperations);
        }
    }

    private static FusedRegionPlan regionWithScaleChain(int operationCount) {
        MatrixExpr expression = MatrixExpr.input(new Matrix(1, 1));
        for (int index = 0; index < operationCount; index++) {
            expression = expression.scale(1.25 + index * 0.001);
        }
        CompiledMatrixProgram compiled = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        assertEquals(1, compiled.cpuPlan().fusedRegions().size());
        return compiled.cpuPlan().fusedRegions().get(0);
    }

    private static long medianMicros(long[] samples) {
        java.util.Arrays.sort(samples);
        return samples[samples.length / 2] / 1_000L;
    }

    private record Row(int operations,
                       long loweringMicros,
                       long verificationMicros,
                       int irValues,
                       int irOperations) {
    }
}
