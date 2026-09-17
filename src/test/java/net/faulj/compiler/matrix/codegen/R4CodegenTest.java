package net.faulj.compiler.matrix.codegen;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.kernel.KernelLowerer;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;
import net.faulj.compiler.matrix.kernel.KernelAliasFact;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOpcode;
import net.faulj.compiler.matrix.affine.AliasRelation;
import net.faulj.matrix.Matrix;

import org.junit.Test;

/** Structural R4 proof for deterministic planning and source emission. */
public class R4CodegenTest {
    @Test
    public void signatureAndSourcesAreDeterministicAndStrict() throws Exception {
        KernelLoweringResult lowering = lower(MatrixExpr.input(matrix(4, 5, 1.0))
            .scale(2.0).add(MatrixExpr.input(matrix(4, 5, 3.0))));
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());

        assertEquals(plan.eligibility().toString(),
            SimdEligibility.Status.AVX2_CONTIGUOUS, plan.eligibility().status());
        GeneratedKernelSource scalarFirst = KernelCodeGenerator.scalar(plan);
        GeneratedKernelSource scalarSecond = KernelCodeGenerator.scalar(plan);
        GeneratedKernelSource avxFirst = KernelCodeGenerator.avx2(plan);
        GeneratedKernelSource avxSecond = KernelCodeGenerator.avx2(plan);

        assertEquals(plan.signature(), KernelSignature.from(plan.function()));
        assertEquals(scalarFirst.source(), scalarSecond.source());
        assertEquals(avxFirst.source(), avxSecond.source());
        assertTrue(scalarFirst.source().contains(" * "));
        assertTrue(avxFirst.source().contains("_mm256_mul_pd"));
        assertTrue(avxFirst.source().contains("_mm256_add_pd"));
        assertTrue(avxFirst.source().contains("_mm256_loadu_pd"));
        assertTrue(avxFirst.source().contains("_mm256_storeu_pd"));
        assertFalse(avxFirst.source().contains("fmadd"));
        assertTrue(avxFirst.source().indexOf("_mm256_set1_pd(2.0)")
            < avxFirst.source().indexOf("for (; p + 4 <= count"));
        assertEquals(4, plan.vectorWidth());
        assertEquals(5, plan.vectorIterations());
        assertEquals(0, plan.scalarTailElements());
        assertEquals(24L, plan.estimatedBytesPerElement());
        writeInspectionReport(plan, scalarFirst, avxFirst);
    }

    @Test
    public void transposeIsScalarCppOnlyAndAliasIsNeverPromoted() {
        KernelLoweringResult transpose = lower(MatrixExpr.input(matrix(5, 4, 1.0)).transpose()
            .scale(2.0).add(MatrixExpr.input(matrix(4, 5, 3.0))));
        PseudokernelPlan transposePlan = PseudokernelPlanner.plan(transpose.program().function());
        assertEquals(SimdEligibility.Status.SCALAR_CPP, transposePlan.eligibility().status());
        assertTrue(transposePlan.eligibility().reason().contains("non-unit-stride"));
        assertTrue(KernelCodeGenerator.scalar(transposePlan).source().contains("for (std::size_t i"));

        KernelLoweringResult contiguous = lower(MatrixExpr.input(matrix(3, 5, 1.0)).scale(2.0)
            .add(MatrixExpr.input(matrix(3, 5, 4.0))));
        KernelFunction function = contiguous.program().function();
        var aliasFacts = function.aliasFacts().stream().map(fact ->
            fact.second() == function.outputBuffers().get(0)
                ? new KernelAliasFact(fact.first(), fact.second(), AliasRelation.MAY_ALIAS)
                : fact).toList();
        KernelFunction conservative = new KernelFunction(
            function.id(), function.name(), function.inputBuffers(), function.outputBuffers(),
            function.iterationDomain(), function.loops(), function.body(), function.values(),
            function.provenance(), aliasFacts);
        assertEquals(SimdEligibility.Status.SCALAR_CPP,
            PseudokernelPlanner.plan(conservative).eligibility().status());
    }

    @Test
    public void sharedSsaAndMetricsRemainShared() {
        Matrix a = matrix(2, 6, 1.0);
        MatrixExpr shared = MatrixExpr.input(a).scale(3.0);
        KernelLoweringResult lowering = lower(shared.add(shared).add(shared));
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
        long muls = lowering.program().function().body().operations().stream()
            .filter(operation -> operation.opcode() == KernelOpcode.MUL).count();
        assertEquals(1L, muls);
        assertTrue(plan.maxLiveVectorValues() >= 1);
    }

    @Test
    public void vectorPlanAccountsForEveryTailRemainderAndZeroSize() {
        for (int columns = 1; columns <= 5; columns++) {
            Matrix a = matrix(1, columns, 1.0);
            Matrix b = matrix(1, columns, 2.0);
            KernelLoweringResult lowering = lower(
                MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
            PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
            assertEquals(columns / 4, plan.vectorIterations());
            assertEquals(columns % 4, plan.scalarTailElements());
            assertTrue(KernelCodeGenerator.avx2(plan).source().contains("for (; p < count"));
        }
    }

    @Test
    public void generatedBackendMissFallsBackWithoutChangingDefault() {
        String previous = System.getProperty(KernelBackendMode.PROPERTY);
        try {
            System.setProperty(KernelBackendMode.PROPERTY, "avx2");
            CompiledMatrixProgram program = MatrixCompiler.compileProgram(
                MatrixExpr.input(matrix(3, 5, 1.0)).scale(2.0),
                OptimizationSemantics.STRICT, new FlopCostModel(), FusionStrategy.GENERALIZED);
            assertEquals(1, program.cpuPlan().stepCount());
            assertEquals(15, program.execute().getRawData().length);
            assertEquals(2.0, program.execute().get(0, 0), 0.0);
        } finally {
            if (previous == null) {
                System.clearProperty(KernelBackendMode.PROPERTY);
            } else {
                System.setProperty(KernelBackendMode.PROPERTY, previous);
            }
        }
    }

    private static KernelLoweringResult lower(MatrixExpr expression) {
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        return KernelLowerer.lower(program.cpuPlan().fusedRegions().get(0));
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + index;
        }
        return Matrix.wrap(data, rows, columns);
    }

    private static void writeInspectionReport(PseudokernelPlan plan,
                                              GeneratedKernelSource scalar,
                                              GeneratedKernelSource avx2) throws Exception {
        Path report = Path.of("build", "reports", "r4", "generated");
        Files.createDirectories(report);
        Files.writeString(report.resolve(plan.signature().generatedSymbol() + "_scalar.cpp"),
            scalar.source(), StandardCharsets.UTF_8);
        Files.writeString(report.resolve(plan.signature().generatedSymbol() + "_avx2.cpp"),
            avx2.source(), StandardCharsets.UTF_8);
        Files.writeString(report.resolve(plan.signature().generatedSymbol() + "_signature.txt"),
            plan.signature().canonicalText() + "\n\n" + plan.diagnostic(),
            StandardCharsets.UTF_8);
    }
}
