package net.faulj.compiler.matrix.kernel;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.CpuFusedRegionStep;
import net.faulj.compiler.matrix.cpu.CpuStepKind;
import net.faulj.compiler.matrix.cpu.FusedRegionPlan;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.AliasRelation;
import net.faulj.compiler.matrix.affine.BufferKind;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.DependenceGraph;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.matrix.Matrix;

/** Focused R3 coverage for portable Kernel IR, lowering, and verification. */
public class KernelIrTest {
    @Test
    public void simpleAddKernelHasExplicitLoadAddStore() {
        ManualFixture fixture = manualAdd();
        KernelVerificationResult verification = KernelVerifier.verify(fixture.program);

        assertTrue(verification.toString(), verification.valid());
        assertEquals(
            List.of(KernelOpcode.LOAD, KernelOpcode.LOAD, KernelOpcode.ADD, KernelOpcode.STORE),
            fixture.program.function().body().operations().stream()
                .map(KernelOp::opcode).toList());
        Matrix actual = KernelReferenceExecutor.execute(
            fixture.program, fixture.bindingMatrices());
        assertArrayEquals(new double[]{6, 8, 10, 12}, actual.getRawData(), 0.0);
    }

    @Test
    public void scaleKernelUsesConstantAndMul() {
        ManualFixture fixture = manualScale();
        assertTrue(KernelVerifier.verify(fixture.program).valid());
        assertEquals(
            List.of(KernelOpcode.LOAD, KernelOpcode.CONSTANT, KernelOpcode.MUL,
                KernelOpcode.STORE),
            fixture.program.function().body().operations().stream()
                .map(KernelOp::opcode).toList());
        Matrix actual = KernelReferenceExecutor.execute(
            fixture.program, fixture.bindingMatrices());
        assertArrayEquals(new double[]{2, 4, 6, 8}, actual.getRawData(), 0.0);
    }

    @Test
    public void scaleAddLoweringIsDistinctFromR2ScalarProgram() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
        KernelLoweringResult lowering = lower(compiled);
        KernelFunction function = lowering.program().function();

        assertTrue(lowering.isEligible());
        assertEquals(
            List.of(KernelOpcode.LOAD, KernelOpcode.CONSTANT, KernelOpcode.MUL,
                KernelOpcode.LOAD, KernelOpcode.ADD, KernelOpcode.STORE),
            function.body().operations().stream().map(KernelOp::opcode).toList());
        assertEquals(1L, function.body().operations().stream()
            .filter(operation -> operation.opcode() == KernelOpcode.STORE).count());
        assertTrue(function.dump().contains("fused-region=F"));
        assertTrue(function.dump().contains("eliminated=[%"));
        assertTrue(compiled.cpuPlan().fusedRegions().get(0).scalarProgram().nodeCount() > 0);
    }

    @Test
    public void longTenPlusNodeChainRemainsSharedAndDeterministic() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        MatrixExpr expression = MatrixExpr.input(a);
        for (int index = 0; index < 12; index++) {
            expression = expression.scale(1.25 + index * 0.125);
        }
        CompiledMatrixProgram compiled = generalized(expression);
        KernelLoweringResult first = lower(compiled);
        KernelLoweringResult second = lower(compiled);

        assertTrue(first.isEligible());
        assertEquals(first.program().dump(), second.program().dump());
        assertEquals(12, compiled.cpuPlan().fusedRegions().get(0).statements().size());
        assertTrue(first.program().function().body().operations().size() > 12);
        assertArrayEquals(compiled.execute().getRawData(),
            executeLowered(compiled, first).getRawData(), 0.0);
    }

    @Test
    public void sharedProducerWithThreeUsesIsOneKernelValue() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        MatrixExpr shared = MatrixExpr.input(a).scale(3.0);
        CompiledMatrixProgram compiled = generalized(
            shared.add(shared).add(shared));
        KernelLoweringResult lowering = lower(compiled);
        KernelFunction function = lowering.program().function();

        assertEquals(1, compiled.cpuPlan().fusedRegions().get(0)
            .scalarProgram().sharedProducerCount());
        KernelOp sharedMul = function.body().operations().stream()
            .filter(operation -> operation.opcode() == KernelOpcode.MUL)
            .findFirst().orElseThrow();
        long uses = function.body().operations().stream()
            .flatMap(operation -> operation.operands() == null
                ? java.util.stream.Stream.<Integer>empty() : operation.operands().stream())
            .filter(operand -> operand == sharedMul.resultValueId())
            .count();
        assertEquals(3L, uses);
        assertEquals(1, KernelIrMetrics.from(List.of(lowering)).sharedValueCount());
    }

    @Test
    public void transposeIsAnAffineLoadMapNotAnOpcode() {
        Matrix a = matrix(37, 11, sequence(37 * 11));
        Matrix b = matrix(11, 37, sequence(11 * 37, 1000));
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).transpose().scale(2.0).add(MatrixExpr.input(b)));
        KernelLoweringResult lowering = lower(compiled);
        KernelFunction function = lowering.program().function();

        assertEquals(0, function.body().operations().stream()
            .filter(operation -> operation.opcode().name().equals("TRANSPOSE")).count());
        KernelOp loadA = function.body().operations().stream()
            .filter(operation -> operation.opcode() == KernelOpcode.LOAD
                && operation.access().buffer().logicalBuffer().name().equals("input0"))
            .findFirst().orElseThrow();
        assertEquals(List.of(AffineExpr.variable(new AffineVariable("j")),
            AffineExpr.variable(new AffineVariable("i"))), loadA.access().indices());
        assertArrayEquals(
            a.transpose().multiplyScalar(2.0).add(b).getRawData(),
            executeLowered(compiled, lowering).getRawData(), 0.0);
    }

    @Test
    public void doubleTransposeCompositionAndRectangularShapesExecute() {
        for (int[] shape : new int[][]{{37, 11}, {11, 37}}) {
            Matrix a = matrix(shape[0], shape[1], sequence(shape[0] * shape[1]));
            Matrix b = matrix(shape[0], shape[1], sequence(shape[0] * shape[1], 100));
            MatrixExpr expression = MatrixExpr.input(a).transpose().scale(2.0)
                .transpose().add(MatrixExpr.input(b));
            CompiledMatrixProgram compiled = generalized(expression);
            KernelLoweringResult lowering = lower(compiled);

            assertTrue(lowering.isEligible());
            assertArrayEquals(
                a.transpose().multiplyScalar(2.0).transpose().add(b).getRawData(),
                executeLowered(compiled, lowering).getRawData(), 0.0);
            assertEquals(0, lowering.program().function().body().operations().stream()
                .filter(operation -> operation.opcode().name().equals("TRANSPOSE")).count());
        }
    }

    @Test
    public void edgeDimensionsAndSpecialRealValuesMatchExactly() {
        Matrix zeroA = new Matrix(0, 3);
        Matrix zeroB = new Matrix(0, 3);
        CompiledMatrixProgram zero = generalized(
            MatrixExpr.input(zeroA).scale(2.0).add(MatrixExpr.input(zeroB)));
        assertEquals(0, executeLowered(zero, lower(zero)).getRawData().length);

        for (double first : new double[]{Double.NaN, Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY, 0.0, -0.0}) {
            Matrix a = matrix(1, 1, first);
            Matrix b = matrix(1, 1, -0.0);
            CompiledMatrixProgram compiled = generalized(
                MatrixExpr.input(a).scale(-1.0).add(MatrixExpr.input(b)));
            Matrix expected = a.multiplyScalar(-1.0).add(b);
            Matrix actual = executeLowered(compiled, lower(compiled));
            assertEquals(Double.doubleToRawLongBits(expected.get(0, 0)),
                Double.doubleToRawLongBits(actual.get(0, 0)));
        }
    }

    @Test
    public void tiledR2ScheduleDoesNotProduceAnUnsafeKernel() {
        Matrix a = matrix(5, 3, sequence(15));
        Matrix b = matrix(5, 3, sequence(15, 50));
        var expressionPlan = MatrixCompiler.compile(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
        AffineProgram affine = AffineProgram.lower(expressionPlan);
        SchedulePlan tiled = SchedulePlan.initial(affine)
            .stripMine(1, "i", 2).schedule();
        var cpuPlan = net.faulj.compiler.matrix.cpu.CpuLowerer.lower(
            tiled, expressionPlan, FusionStrategy.GENERALIZED);
        CpuFusedRegionStep fused = (CpuFusedRegionStep) cpuPlan.steps().get(0);
        KernelLoweringResult lowering = KernelLowerer.lower(fused.regionPlan());

        assertEquals(KernelEligibility.INELIGIBLE, lowering.eligibility());
        assertNull(lowering.program());
        assertArrayEquals(a.multiplyScalar(2.0).add(b).getRawData(),
            cpuPlan.execute().getRawData(), 0.0);
    }

    @Test
    public void strictOperationAssociationIsKept() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).add(MatrixExpr.input(b)).scale(0.25));
        List<KernelOpcode> opcodes = lower(compiled).program().function().body().operations()
            .stream().map(KernelOp::opcode).toList();
        assertEquals(List.of(KernelOpcode.LOAD, KernelOpcode.LOAD, KernelOpcode.ADD,
            KernelOpcode.CONSTANT, KernelOpcode.MUL, KernelOpcode.STORE), opcodes);
        assertArrayEquals(
            a.add(b).multiplyScalar(0.25).getRawData(),
            executeLowered(compiled, lower(compiled)).getRawData(), 0.0);
    }

    @Test
    public void complexRuntimeStorageIsExplicitlyIneligibleAndR2StillRuns() {
        Matrix complex = new Matrix(
            new double[][]{{1, 2}, {3, 4}},
            new double[][]{{0.5, -0.0}, {Double.NaN, 2.0}});
        Matrix real = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(complex).scale(2.0).add(MatrixExpr.input(real)));
        FusedRegionPlan region = compiled.cpuPlan().fusedRegions().get(0);
        IdentityHashMap<LogicalBuffer, Matrix> bindings = logicalBindings(compiled);
        bindings.put(region.outputBuffer(), new Matrix(2, 2));
        KernelLoweringResult lowering = KernelLowerer.lower(region, bindings);

        assertEquals(KernelEligibility.INELIGIBLE, lowering.eligibility());
        assertTrue(lowering.reason().contains("complex"));
        assertArrayEquals(compiled.execute().getRawData(),
            complex.multiplyScalar(2.0).add(real).getRawData(), 0.0);
        assertTrue(compiled.execute().hasImagData());
    }

    @Test
    public void invalidSsaReferenceIsRejectedWithDeterministicDiagnostic() {
        ManualFixture fixture = manualAdd();
        List<KernelOp> operations = new ArrayList<>(fixture.function.body().operations());
        operations.set(2, KernelOp.add(2, 0, 99, 1, 2));
        KernelFunction invalid = fixture.withBody(operations).function;

        KernelVerificationResult result = KernelVerifier.verify(invalid);
        assertFalse(result.valid());
        assertTrue(result.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("undefined or non-dominating SSA operand")));
    }

    @Test
    public void duplicateValueIdIsRejected() {
        ManualFixture fixture = manualAdd();
        List<KernelValue> values = new ArrayList<>(fixture.function.values());
        values.add(new KernelValue(0, KernelValueType.FP64, "duplicate"));
        KernelFunction invalid = fixture.withValues(values).function;

        KernelVerificationResult result = KernelVerifier.verify(invalid);
        assertFalse(result.valid());
        assertTrue(result.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("duplicate SSA value ID")));
    }

    @Test
    public void undefinedLoopVariableIsRejected() {
        ManualFixture fixture = manualAdd();
        KernelAccess invalidAccess = new KernelAccess(
            fixture.a, List.of(AffineExpr.variable(new AffineVariable("k")),
                AffineExpr.variable(fixture.i)));
        List<KernelOp> operations = new ArrayList<>(fixture.function.body().operations());
        operations.set(0, KernelOp.load(0, invalidAccess, -1, 0));
        KernelFunction invalid = fixture.withBody(operations).function;

        KernelVerificationResult result = KernelVerifier.verify(invalid);
        assertFalse(result.valid());
        assertTrue(result.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("undefined loop variable")));
    }

    @Test
    public void illegalStoreToInputIsRejected() {
        ManualFixture fixture = manualAdd();
        List<KernelOp> operations = new ArrayList<>(fixture.function.body().operations());
        operations.set(3, KernelOp.store(new KernelAccess(
            fixture.a, List.of(AffineExpr.variable(fixture.i),
                AffineExpr.variable(fixture.j))), 2, 1, 2));
        KernelFunction invalid = fixture.withBody(operations).function;

        KernelVerificationResult result = KernelVerifier.verify(invalid);
        assertFalse(result.valid());
        assertTrue(result.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("STORE target is not a writable OUTPUT")));
    }

    @Test
    public void shapeRankMismatchAndUnsupportedAffineMapAreRejected() {
        ManualFixture fixture = manualAdd();
        List<KernelOp> rankOperations = new ArrayList<>(fixture.function.body().operations());
        rankOperations.set(0, KernelOp.load(0,
            new KernelAccess(fixture.a, List.of(AffineExpr.variable(fixture.i))), -1, 0));
        KernelVerificationResult rank = KernelVerifier.verify(fixture.withBody(rankOperations).function);
        assertFalse(rank.valid());
        assertTrue(rank.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("access rank")));

        List<KernelOp> affineOperations = new ArrayList<>(fixture.function.body().operations());
        affineOperations.set(0, KernelOp.load(0, new KernelAccess(
            fixture.a, List.of(AffineExpr.variable(fixture.i).scale(2),
                AffineExpr.variable(fixture.j))), -1, 0));
        KernelVerificationResult affine = KernelVerifier.verify(
            fixture.withBody(affineOperations).function);
        assertFalse(affine.valid());
        assertTrue(affine.diagnostics().stream()
            .anyMatch(diagnostic -> diagnostic.contains("unsupported affine access expression")));
    }

    @Test
    public void aliasFactsAreConservativeAndPreserved() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
        KernelFunction function = lower(compiled).program().function();
        KernelBuffer first = function.inputBuffers().get(0);
        KernelBuffer second = function.inputBuffers().get(1);
        KernelBuffer output = function.outputBuffers().get(0);

        assertEquals(AliasRelation.MAY_ALIAS, function.aliasRelation(first, second));
        assertEquals(AliasRelation.NO_ALIAS, function.aliasRelation(first, output));
        assertEquals(AliasRelation.NO_ALIAS, function.aliasRelation(second, output));

        List<KernelAliasFact> illegalFacts = new ArrayList<>(function.aliasFacts());
        illegalFacts.set(0, new KernelAliasFact(first, second, AliasRelation.NO_ALIAS));
        KernelFunction invalid = new KernelFunction(
            function.id(), function.name(), function.inputBuffers(), function.outputBuffers(),
            function.iterationDomain(), function.loops(), function.body(), function.values(),
            function.provenance(), illegalFacts);
        assertFalse(KernelVerifier.verify(invalid).valid());
    }

    @Test
    public void r1PhysicalSlotBindingStaysOutsideKernelSemantics() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
        FusedRegionPlan region = compiled.cpuPlan().fusedRegions().get(0);
        KernelFunction function = lower(compiled).program().function();
        IdentityHashMap<LogicalBuffer, Matrix> logical = logicalBindings(compiled);
        logical.put(region.outputBuffer(), new Matrix(2, 2));
        KernelBinding binding = KernelBinding.fromLogicalBuffers(
            function, logical, compiled.cpuPlan().physicalMemoryPlan());

        assertNotNull(compiled.cpuPlan().physicalMemoryPlan().slotFor(region.outputBuffer()));
        assertSame(compiled.cpuPlan().physicalMemoryPlan().slotFor(region.outputBuffer()),
            binding.physicalSlot(function.outputBuffers().get(0)));
        assertTrue(function.inputBuffers().stream()
            .allMatch(input -> binding.physicalSlot(input) == null));
        assertFalse(function.dump().contains("Matrix@"));
    }

    @Test
    public void provenanceAndCanonicalDumpAreStable() {
        Matrix a = matrix(1, 1, 1.0);
        Matrix b = matrix(1, 1, 2.0);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
        KernelProgram program = lower(compiled).program();
        String first = program.dump();
        String second = lower(compiled).program().dump();

        assertEquals(first, second);
        assertTrue(first.contains("source:"));
        assertTrue(first.contains("statements=[S"));
        assertTrue(first.contains("legality="));
        assertTrue(first.contains("profitability="));
        assertTrue(first.contains("loops:"));
        assertTrue(first.contains("body:"));
        assertTrue(first.contains("store %"));
        assertFalse(first.contains("TRANSPOSE"));
    }

    @Test
    public void escapingR2RegionIsRejectedBeforeKernelConstructionAndGemmStaysOpaque() {
        Matrix a = matrix(2, 2, 1, 2, 3, 4);
        Matrix b = matrix(2, 2, 5, 6, 7, 8);
        Matrix c = matrix(2, 2, 2, 1, 4, 3);
        CompiledMatrixProgram compiled = generalized(
            MatrixExpr.input(a).scale(2.0).matmul(MatrixExpr.input(b))
                .add(MatrixExpr.input(c)));

        assertTrue(compiled.cpuPlan().steps().stream()
            .anyMatch(step -> step.kind() == CpuStepKind.GEMM));
        assertTrue(compiled.cpuPlan().fusedRegions().isEmpty());
        assertTrue(compiled.cpuPlan().kernelPrograms().isEmpty());
        assertArrayEquals(
            a.multiplyScalar(2.0).multiply(b).add(c).getRawData(),
            compiled.execute().getRawData(), 0.0);
    }

    @Test
    public void optionalExecuteModeUsesKernelReferenceAndVerifyModeKeepsR2Path() {
        String previous = System.getProperty(KernelIrMode.PROPERTY);
        try {
            System.setProperty(KernelIrMode.PROPERTY, "execute");
            Matrix a = matrix(2, 2, 1, 2, 3, 4);
            Matrix b = matrix(2, 2, 5, 6, 7, 8);
            CompiledMatrixProgram execute = generalized(
                MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
            Matrix actual = execute.execute();
            CpuFusedRegionStep executeStep =
                (CpuFusedRegionStep) execute.cpuPlan().steps().get(0);
            assertEquals(CpuFusedRegionStep.ExecutionPath.KERNEL_IR_REFERENCE,
                executeStep.lastExecutionPath());
            assertArrayEquals(a.multiplyScalar(2.0).add(b).getRawData(),
                actual.getRawData(), 0.0);
            assertEquals(1, execute.cpuPlan().kernelIrMetrics().kernelCount());

            System.setProperty(KernelIrMode.PROPERTY, "verify");
            CompiledMatrixProgram verify = generalized(
                MatrixExpr.input(a).scale(2.0).add(MatrixExpr.input(b)));
            CpuFusedRegionStep verifyStep =
                (CpuFusedRegionStep) verify.cpuPlan().steps().get(0);
            assertNotNull(verifyStep.kernelLowering());
            verify.execute();
            assertEquals(CpuFusedRegionStep.ExecutionPath.FAST_REAL_DENSE,
                verifyStep.lastExecutionPath());
        } finally {
            if (previous == null) {
                System.clearProperty(KernelIrMode.PROPERTY);
            } else {
                System.setProperty(KernelIrMode.PROPERTY, previous);
            }
        }
    }

    private static CompiledMatrixProgram generalized(MatrixExpr expression) {
        return MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
    }

    private static KernelLoweringResult lower(CompiledMatrixProgram compiled) {
        assertEquals(1, compiled.cpuPlan().fusedRegions().size());
        return KernelLowerer.lower(compiled.cpuPlan().fusedRegions().get(0));
    }

    private static Matrix executeLowered(CompiledMatrixProgram compiled,
                                         KernelLoweringResult lowering) {
        assertTrue(lowering.isEligible());
        FusedRegionPlan region = compiled.cpuPlan().fusedRegions().get(0);
        IdentityHashMap<LogicalBuffer, Matrix> logical = logicalBindings(compiled);
        logical.put(region.outputBuffer(), new Matrix(
            region.outputBuffer().shape().rows(), region.outputBuffer().shape().columns()));
        KernelBinding binding = KernelBinding.fromLogicalBuffers(
            lowering.program().function(), logical, compiled.cpuPlan().physicalMemoryPlan());
        return KernelReferenceExecutor.executeVerified(lowering.program(), binding);
    }

    private static IdentityHashMap<LogicalBuffer, Matrix> logicalBindings(
        CompiledMatrixProgram compiled) {
        IdentityHashMap<LogicalBuffer, Matrix> result = new IdentityHashMap<>();
        for (net.faulj.compiler.matrix.cpu.CpuBufferBinding binding
                 : compiled.cpuPlan().inputBindings()) {
            if (binding.matrix() != null) {
                result.put(binding.buffer(), binding.matrix());
            }
        }
        return result;
    }

    private static ManualFixture manualAdd() {
        return manual(
            List.of(
                KernelOp.load(0, null, -1, 0),
                KernelOp.load(1, null, -1, 1),
                KernelOp.add(2, 0, 1, 1, 2),
                KernelOp.store(null, 2, 1, 2)),
            List.of(new KernelValue(0, KernelValueType.FP64),
                new KernelValue(1, KernelValueType.FP64),
                new KernelValue(2, KernelValueType.FP64)),
            true);
    }

    private static ManualFixture manualScale() {
        return manual(
            List.of(
                KernelOp.load(0, null, -1, 0),
                KernelOp.constant(1, 2.0, 1, 1),
                KernelOp.mul(2, 1, 0, 1, 1),
                KernelOp.store(null, 2, 1, 2)),
            List.of(new KernelValue(0, KernelValueType.FP64),
                new KernelValue(1, KernelValueType.FP64),
                new KernelValue(2, KernelValueType.FP64)),
            false);
    }

    private static ManualFixture manual(List<KernelOp> operationTemplate,
                                        List<KernelValue> values,
                                        boolean add) {
        MatrixShape shape = new MatrixShape(2, 2);
        LogicalBuffer aLogical = external(0, "A", shape);
        LogicalBuffer bLogical = external(1, "B", shape);
        LogicalBuffer outputLogical = temporary(2, "C", shape);
        KernelBuffer a = KernelBuffer.fromLogical(aLogical, KernelBufferRole.INPUT);
        KernelBuffer b = KernelBuffer.fromLogical(bLogical, KernelBufferRole.INPUT);
        KernelBuffer output = KernelBuffer.fromLogical(outputLogical, KernelBufferRole.OUTPUT);
        AffineVariable i = new AffineVariable("i");
        AffineVariable j = new AffineVariable("j");
        KernelAccess aAccess = new KernelAccess(a, List.of(
            AffineExpr.variable(i), AffineExpr.variable(j)));
        KernelAccess bAccess = new KernelAccess(b, List.of(
            AffineExpr.variable(i), AffineExpr.variable(j)));
        KernelAccess outputAccess = new KernelAccess(output, List.of(
            AffineExpr.variable(i), AffineExpr.variable(j)));
        List<KernelOp> operations = new ArrayList<>();
        for (KernelOp operation : operationTemplate) {
            if (operation.opcode() == KernelOpcode.LOAD) {
                KernelBuffer buffer = operationTemplate.indexOf(operation) == 0 ? a : b;
                operations.add(KernelOp.load(operation.resultValueId(),
                    buffer == a ? aAccess : bAccess,
                    operation.sourceStatementId(), operation.sourceScalarValueId()));
            } else if (operation.opcode() == KernelOpcode.STORE) {
                operations.add(KernelOp.store(outputAccess,
                    operation.operands().get(0), operation.sourceStatementId(),
                    operation.sourceScalarValueId()));
            } else {
                operations.add(operation);
            }
        }
        KernelProvenance provenance = new KernelProvenance(
            0, add ? List.of(0, 1) : List.of(0, 1),
            List.of(0, 1, 2), List.of(1), "manual legality", "manual profitability");
        KernelFunction function = new KernelFunction(
            0, add ? "add" : "scale", List.of(a, b), List.of(output),
            IterationDomain.of(
                IterationDomain.range(i, 0, 2), IterationDomain.range(j, 0, 2)),
            List.of(new KernelLoop(i, 0, 2), new KernelLoop(j, 0, 2)),
            new KernelBlock(operations), values, provenance);
        return new ManualFixture(
            KernelProgram.single("manual", function), function, a, b, output,
            i, j, aLogical, bLogical, outputLogical);
    }

    private record ManualFixture(KernelProgram program,
                                 KernelFunction function,
                                 KernelBuffer a,
                                 KernelBuffer b,
                                 KernelBuffer output,
                                 AffineVariable i,
                                 AffineVariable j,
                                 LogicalBuffer aLogical,
                                 LogicalBuffer bLogical,
                                 LogicalBuffer outputLogical) {
        private Map<KernelBuffer, Matrix> bindingMatrices() {
            IdentityHashMap<KernelBuffer, Matrix> result = new IdentityHashMap<>();
            result.put(a, matrix(2, 2, 1, 2, 3, 4));
            result.put(b, matrix(2, 2, 5, 6, 7, 8));
            result.put(output, new Matrix(2, 2));
            return result;
        }

        private ManualFixture withBody(List<KernelOp> operations) {
            KernelFunction updated = new KernelFunction(
                function.id(), function.name(), function.inputBuffers(), function.outputBuffers(),
                function.iterationDomain(), function.loops(), new KernelBlock(operations),
                function.values(), function.provenance(), function.aliasFacts());
            return new ManualFixture(program, updated, a, b, output, i, j,
                aLogical, bLogical, outputLogical);
        }

        private ManualFixture withValues(List<KernelValue> values) {
            KernelFunction updated = new KernelFunction(
                function.id(), function.name(), function.inputBuffers(), function.outputBuffers(),
                function.iterationDomain(), function.loops(), function.body(), values,
                function.provenance(), function.aliasFacts());
            return new ManualFixture(program, updated, a, b, output, i, j,
                aLogical, bLogical, outputLogical);
        }
    }

    private static LogicalBuffer external(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.EXTERNAL_INPUT,
            BufferOwnership.BORROWED, MemorySpace.HEAP, shape);
    }

    private static LogicalBuffer temporary(int id, String name, MatrixShape shape) {
        return new LogicalBuffer(id, name, BufferKind.TEMPORARY,
            BufferOwnership.OWNED, MemorySpace.UNKNOWN, shape);
    }

    private static Matrix matrix(int rows, int columns, double... values) {
        return Matrix.wrap(values.clone(), rows, columns);
    }

    private static double[] sequence(int length) {
        return sequence(length, 0);
    }

    private static double[] sequence(int length, int offset) {
        double[] result = new double[length];
        for (int index = 0; index < result.length; index++) {
            result[index] = index + offset;
        }
        return result;
    }
}
