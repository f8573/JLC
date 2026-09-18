package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.cpu.FusedRegionPlan;
import net.faulj.compiler.matrix.cpu.ScalarFusionProgram;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.matrix.Matrix;

/** Dedicated deterministic lowering from the R2 fused-region representation. */
public final class KernelLowerer {
    private KernelLowerer() {
    }

    /** Lower a legal R2 region using only compile-time logical storage facts. */
    public static KernelLoweringResult lower(FusedRegionPlan region) {
        return lowerInternal(region, null, false);
    }

    /**
     * Lower a region and classify runtime lane eligibility from borrowed
     * matrix bindings. The matrices are inspected for facts only and are not
     * retained by the resulting immutable Kernel IR.
     */
    public static KernelLoweringResult lower(FusedRegionPlan region,
                                             Map<LogicalBuffer, Matrix> runtimeBindings) {
        if (runtimeBindings == null) {
            return lowerInternal(region, null, true);
        }
        return lowerInternal(region, runtimeBindings, true);
    }

    private static KernelLoweringResult lowerInternal(FusedRegionPlan region,
                                                       Map<LogicalBuffer, Matrix> runtimeBindings,
                                                       boolean classifyRuntimeStorage) {
        long loweringStart = System.nanoTime();
        KernelProgram program = null;
        try {
            KernelFunction function = buildFunction(region);
            program = KernelProgram.single("kernel-program-F" + region.regionId(), function);
        } catch (UnsupportedKernel unsupported) {
            long loweringTime = elapsed(loweringStart);
            return new KernelLoweringResult(
                KernelEligibility.INELIGIBLE, null, null, unsupported.getMessage(),
                loweringTime, 0L);
        } catch (RuntimeException failure) {
            long loweringTime = elapsed(loweringStart);
            return new KernelLoweringResult(
                KernelEligibility.INELIGIBLE, null, null,
                "lowering failed: " + message(failure), loweringTime, 0L);
        }
        long loweringTime = elapsed(loweringStart);

        long verificationStart = System.nanoTime();
        KernelVerificationResult verification = KernelVerifier.verify(program);
        long verificationTime = elapsed(verificationStart);
        if (!verification.valid()) {
            return new KernelLoweringResult(
                KernelEligibility.INELIGIBLE, program, verification,
                "verification failed: " + verification.diagnostic(),
                loweringTime, verificationTime);
        }

        KernelEligibility eligibility = KernelEligibility.ELIGIBLE;
        String reason = "verified Kernel IR";
        if (classifyRuntimeStorage) {
            if (runtimeBindings == null) {
                eligibility = KernelEligibility.UNKNOWN;
                reason = "real FP64 runtime storage was not supplied";
            } else {
                StorageClassification classification = classifyStorage(
                    program.function(), runtimeBindings);
                eligibility = classification.eligibility;
                reason = classification.reason;
            }
        }
        return new KernelLoweringResult(
            eligibility, program, verification, reason, loweringTime, verificationTime);
    }

    private static KernelFunction buildFunction(FusedRegionPlan region) {
        if (region == null) {
            throw reject("R2 fused region is null");
        }
        if (region.legality() != LegalityStatus.LEGAL) {
            throw reject("R2 region is not proven legal");
        }
        if (region.outputBuffer() == null || !region.outputBuffer().isTemporary()) {
            throw reject("R2 output is not a writable temporary");
        }
        validateCanonicalLoops(region);

        IdentityHashMap<LogicalBuffer, KernelBuffer> inputsByLogical = new IdentityHashMap<>();
        List<KernelBuffer> inputs = new ArrayList<>();
        for (LogicalBuffer logical : region.leafBuffers()) {
            if (logical == null || inputsByLogical.containsKey(logical)) {
                throw reject("R2 leaf buffers are not unique");
            }
            KernelBuffer input = KernelBuffer.fromLogical(logical, KernelBufferRole.INPUT);
            inputs.add(input);
            inputsByLogical.put(logical, input);
        }
        KernelBuffer output = KernelBuffer.fromLogical(
            region.outputBuffer(), KernelBufferRole.OUTPUT);
        if (inputsByLogical.containsKey(region.outputBuffer())) {
            throw reject("R2 output is also a leaf input");
        }

        List<KernelLoop> loops = new ArrayList<>();
        for (net.faulj.compiler.matrix.schedule.ScheduleLoop loop
                 : region.scheduleBand().loops()) {
            loops.add(new KernelLoop(
                loop.inductionVariable(), loop.semanticVariable(), loop.lowerBound(),
                loop.upperBound(), loop.step(), loop.valueExpression()));
        }

        ScalarFusionProgram scalar = region.scalarProgram();
        if (scalar == null || scalar.nodes().isEmpty()) {
            throw reject("R2 scalar program is empty");
        }
        int[] loweredValues = new int[scalar.nodeCount()];
        Arrays.fill(loweredValues, -1);
        List<KernelValue> values = new ArrayList<>();
        List<KernelOp> operations = new ArrayList<>();
        for (ScalarFusionProgram.ScalarNode node : scalar.nodes()) {
            if (node == null) {
                throw reject("R2 scalar program contains a null node");
            }
            for (Integer input : node.inputs()) {
                if (input == null || input < 0 || input >= loweredValues.length
                    || loweredValues[input] < 0) {
                    throw reject("R2 scalar input is not topologically available at v" + node.id());
                }
            }
            int result;
            switch (node.opcode()) {
                case LOAD -> {
                    KernelBuffer input = inputsByLogical.get(node.buffer());
                    if (input == null) {
                        throw reject("R2 LOAD v" + node.id()
                            + " does not name a leaf buffer");
                    }
                    KernelAccess access = new KernelAccess(input, node.indices());
                    requireUnitAccess(access, "R2 LOAD v" + node.id());
                    result = values.size();
                    values.add(new KernelValue(result, KernelValueType.FP64));
                    operations.add(KernelOp.load(
                        result, access, node.statementId(), node.id()));
                }
                case SCALE -> {
                    if (node.inputs().size() != 1) {
                        throw reject("R2 SCALE v" + node.id() + " has invalid arity");
                    }
                    int input = loweredValues[node.inputs().get(0)];
                    int constant = values.size();
                    values.add(new KernelValue(constant, KernelValueType.FP64));
                    operations.add(KernelOp.constant(
                        constant, node.factor(), node.statementId(), node.id()));
                    result = values.size();
                    values.add(new KernelValue(result, KernelValueType.FP64));
                    // Keep R2's factor * value operand order exactly.
                    operations.add(KernelOp.mul(
                        result, constant, input, node.statementId(), node.id()));
                }
                case ADD -> {
                    if (node.inputs().size() != 2) {
                        throw reject("R2 ADD v" + node.id() + " has invalid arity");
                    }
                    result = values.size();
                    values.add(new KernelValue(result, KernelValueType.FP64));
                    operations.add(KernelOp.add(
                        result, loweredValues[node.inputs().get(0)],
                        loweredValues[node.inputs().get(1)], node.statementId(), node.id()));
                }
                case TRANSPOSE -> {
                    if (node.inputs().size() != 1) {
                        throw reject("R2 TRANSPOSE v" + node.id() + " has invalid arity");
                    }
                    // R2 already composed the transpose into its LOAD access map.
                    // There is intentionally no KernelOp for this semantic node.
                    result = loweredValues[node.inputs().get(0)];
                }
                default -> throw reject("unsupported R2 scalar opcode at v" + node.id());
            }
            loweredValues[node.id()] = result;
        }

        int root = scalar.rootNodeId();
        if (root < 0 || root >= loweredValues.length || loweredValues[root] < 0) {
            throw reject("R2 scalar root is not available");
        }
        KernelAccess outputAccess = new KernelAccess(output, region.outputAccess().indices());
        requireUnitAccess(outputAccess, "R2 output");
        operations.add(KernelOp.store(
            outputAccess, loweredValues[root],
            region.statements().get(region.statements().size() - 1).id(), root));

        KernelProvenance provenance = new KernelProvenance(
            region.regionId(),
            region.statementIds(),
            scalar.nodes().stream().map(ScalarFusionProgram.ScalarNode::id).toList(),
            region.eliminatedBuffers().stream().map(LogicalBuffer::id).toList(),
            region.legalityExplanation(), region.profitability());
        return new KernelFunction(
            region.regionId(), "F" + region.regionId(), inputs, List.of(output),
            region.iterationDomain(), loops, new KernelBlock(operations), values, provenance);
    }

    private static void validateCanonicalLoops(FusedRegionPlan region) {
        if (region.iterationDomain().ranges().size() != 2
            || region.scheduleBand().loops().size() != 2) {
            throw reject("R3 v1 requires a two-dimensional rectangular loop nest");
        }
        for (net.faulj.compiler.matrix.schedule.ScheduleLoop loop
                 : region.scheduleBand().loops()) {
            if (!loop.annotations().isEmpty() || loop.guard() != null
                || loop.valueExpression() == null
                || !loop.semanticVariable().equals(loop.inductionVariable())
                || !loop.valueExpression().equals(
                    AffineExpr.variable(loop.inductionVariable()))
                || loop.step() != 1L) {
                throw reject("scheduled/tiled loop form is not in the R3 canonical subset");
            }
            if (loop.lowerBound() < 0L || loop.step() != 1L) {
                throw reject("R3 loop bounds require zero-based unit-step traversal");
            }
            net.faulj.compiler.matrix.affine.IterationDomain.Range range
                = region.iterationDomain().ranges().stream()
                    .filter(candidate -> candidate.variable().equals(loop.semanticVariable()))
                    .findFirst().orElse(null);
            if (range == null || range.lowerInclusive() != loop.lowerBound()
                || range.upperExclusive() != loop.upperBound()) {
                throw reject("schedule loop does not match the R2 iteration domain");
            }
        }
    }

    private static void requireUnitAccess(KernelAccess access, String role) {
        if (access.rank() != 2) {
            throw reject(role + " has unsupported access rank " + access.rank());
        }
        for (AffineExpr index : access.indices()) {
            if (!isUnitVariable(index)) {
                throw reject(role + " has unsupported affine map " + index);
            }
        }
    }

    private static StorageClassification classifyStorage(
        KernelFunction function,
        Map<LogicalBuffer, Matrix> runtimeBindings) {
        boolean missing = false;
        for (KernelBuffer buffer : function.buffers()) {
            LogicalBuffer logical = buffer.logicalBuffer();
            Matrix matrix = logical == null ? null : runtimeBindings.get(logical);
            if (matrix == null) {
                missing = true;
            } else if (matrix.hasImagData()) {
                return new StorageClassification(
                    KernelEligibility.INELIGIBLE,
                    "complex FP64 storage is unsupported by R3; fallback to R2");
            }
        }
        if (missing) {
            return new StorageClassification(
                KernelEligibility.UNKNOWN,
                "real FP64 storage was not proven for every kernel buffer");
        }
        return new StorageClassification(
            KernelEligibility.ELIGIBLE,
            "verified Kernel IR; all bound storage is real FP64");
    }

    private static boolean isUnitVariable(AffineExpr expression) {
        return expression != null && expression.constant() == 0L
            && expression.coefficients().size() == 1
            && expression.coefficients().firstEntry().getValue() == 1L;
    }

    private static UnsupportedKernel reject(String message) {
        return new UnsupportedKernel(message);
    }

    private static String message(RuntimeException failure) {
        return failure.getMessage() == null
            ? failure.getClass().getSimpleName() : failure.getMessage();
    }

    private static long elapsed(long start) {
        return Math.max(0L, System.nanoTime() - start);
    }

    private record StorageClassification(KernelEligibility eligibility, String reason) {
    }

    private static final class UnsupportedKernel extends RuntimeException {
        private UnsupportedKernel(String message) {
            super(message);
        }
    }
}
