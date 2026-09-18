package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.PlanAdd;
import net.faulj.compiler.matrix.PlanInput;
import net.faulj.compiler.matrix.PlanMatMul;
import net.faulj.compiler.matrix.PlanNode;
import net.faulj.compiler.matrix.PlanScale;
import net.faulj.compiler.matrix.PlanTranspose;
import net.faulj.compiler.matrix.affine.AccessKind;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.BufferKind;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.compiler.matrix.schedule.SchedulePlan;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;
import net.faulj.compiler.matrix.schedule.ScheduleTransformResult;
import net.faulj.matrix.Matrix;

/**
 * M4 lowering from the immutable M3 schedule to a small executable CPU plan.
 *
 * <p>MatMul is deliberately lowered to one opaque call to the existing GEMM
 * facade. The legacy scale-then-add family remains independently executable;
 * the post-M4 generalized fusion strategy emits inspectable fused regions for
 * legal bounded Add/Scale/Transpose schedules.</p>
 */
public final class CpuLowerer {
    private CpuLowerer() {
    }

    /** Lower an M1 plan using the selected fusion implementation. */
    public static CpuExecutionPlan lower(ExecutionPlan plan) {
        return lower(plan, FusionStrategy.fromSystemProperty());
    }

    /** Lower an M1 plan with an explicit fusion implementation. */
    public static CpuExecutionPlan lower(ExecutionPlan plan, FusionStrategy strategy) {
        if (plan == null) {
            throw new IllegalArgumentException("Execution plan must not be null");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Fusion strategy must not be null");
        }
        AffineProgram affine = AffineProgram.lower(plan);
        return lower(defaultSchedule(affine, strategy), plan, strategy);
    }

    /** Lower an affine program for inspection when no runtime plan is available. */
    public static CpuExecutionPlan lower(AffineProgram program) {
        return lower(program, FusionStrategy.fromSystemProperty());
    }

    /** Lower an affine program with an explicit fusion implementation. */
    public static CpuExecutionPlan lower(AffineProgram program, FusionStrategy strategy) {
        if (program == null) {
            throw new IllegalArgumentException("Affine program must not be null");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Fusion strategy must not be null");
        }
        return lower(defaultSchedule(program, strategy), null, strategy);
    }

    /**
     * Lower an inspectable schedule without captured runtime inputs. The
     * result is inspectable and can be executed only with external bindings.
     */
    public static CpuExecutionPlan lower(net.faulj.compiler.matrix.schedule.SchedulePlan schedule) {
        return lower(schedule, null, FusionStrategy.fromSystemProperty());
    }

    /**
     * Lower a schedule and retain borrowed runtime matrices from the
     * originating M1 plan. The matrices are referenced, not copied.
     */
    public static CpuExecutionPlan lower(SchedulePlan schedule, ExecutionPlan executionPlan) {
        return lower(schedule, executionPlan, FusionStrategy.fromSystemProperty());
    }

    /** Lower a plan-only schedule with an explicit fusion implementation. */
    public static CpuExecutionPlan lower(SchedulePlan schedule, FusionStrategy strategy) {
        return lower(schedule, null, strategy);
    }

    /**
     * Lower a schedule with an explicit fusion implementation. Generalized
     * fusion is selected after scheduling and before executable steps or
     * physical-memory planning are derived.
     */
    public static CpuExecutionPlan lower(SchedulePlan schedule,
                                         ExecutionPlan executionPlan,
                                         FusionStrategy strategy) {
        if (schedule == null) {
            throw new IllegalArgumentException("Schedule plan must not be null");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Fusion strategy must not be null");
        }
        if (executionPlan != null
            && executionPlan.semantics() != schedule.program().semantics()) {
            throw new IllegalArgumentException(
                "Execution-plan and schedule optimization semantics must match");
        }
        if (executionPlan != null && !schedule.program().originatesFrom(executionPlan)) {
            throw new IllegalArgumentException(
                "ExecutionPlan and AffineProgram do not share verified M1/M2 provenance");
        }

        CpuExecutableSubsetValidator.validate(schedule);

        PlanMapping mapping = executionPlan == null
            ? null : PlanMapping.create(executionPlan, schedule.program());
        IdentityHashMap<AffineStatement, PlanNode> statementOwners = mapping == null
            ? new IdentityHashMap<>() : mapping.statementOwners;
        List<CpuStep> steps = new ArrayList<>();
        Set<AffineStatement> emitted = Collections.newSetFromMap(new IdentityHashMap<>());
        Set<LogicalBuffer> elided = Collections.newSetFromMap(new IdentityHashMap<>());
        GeneralizedFusionPlanner.Result generalized = strategy == FusionStrategy.GENERALIZED
            ? GeneralizedFusionPlanner.plan(schedule) : null;
        List<FusedRegionPlan> acceptedRegions = generalized == null
            ? new ArrayList<>() : new ArrayList<>(generalized.acceptedRegions());
        List<String> fusionDecisions = generalized == null
            ? new ArrayList<>() : new ArrayList<>(generalized.decisions());

        for (int regionIndex = 0; regionIndex < schedule.regions().size();) {
            ScheduleRegion region = schedule.region(regionIndex);
            FusedRegionPlan generalizedRegion = generalized == null
                ? null : generalized.startingAt(region);
            if (generalizedRegion != null) {
                steps.add(generalizedRegion.step(steps.size()));
                emitted.addAll(generalizedRegion.statements());
                elided.addAll(generalizedRegion.eliminatedBuffers());
                regionIndex = generalizedRegion.endRegionIndex() + 1;
                continue;
            }

            FusedSpec fused = strategy == FusionStrategy.LEGACY
                ? fusableRegion(region, schedule.program(), statementOwners) : null;
            if (fused != null) {
                steps.add(fused.step(steps.size()));
                emitted.add(fused.scaleStatement);
                emitted.add(fused.addStatement);
                elided.add(fused.eliminatedBuffer);
                fusionDecisions.add("legacy scale-add S" + fused.scaleStatement.id()
                    + " -> S" + fused.addStatement.id() + " accepted");
                regionIndex++;
                continue;
            }

            for (AffineStatement statement : region.statements()) {
                if (emitted.contains(statement)) {
                    continue;
                }
                if (statement.kind() == StatementKind.MATMUL_INIT) {
                    AffineStatement update = matMulUpdateFor(statement, schedule.program());
                    steps.add(gemmStep(steps.size(), statement, update));
                    emitted.add(statement);
                    emitted.add(update);
                } else if (statement.kind() == StatementKind.MATMUL_UPDATE) {
                    throw new IllegalArgumentException(
                        "MATMUL_UPDATE S" + statement.id()
                            + " is scheduled without its initialization statement");
                } else {
                    steps.add(simpleStep(steps.size(), region, statement, statementOwners));
                    emitted.add(statement);
                }
            }
            regionIndex++;
        }

        if (emitted.size() != schedule.program().statements().size()) {
            throw new IllegalArgumentException(
                "CPU schedule did not lower every affine statement exactly once");
        }

        List<LogicalBuffer> temporaries = buffersOfKind(schedule.program(), BufferKind.TEMPORARY);
        List<LogicalBuffer> materialized = new ArrayList<>();
        for (LogicalBuffer buffer : temporaries) {
            if (!elided.contains(buffer) && steps.stream().anyMatch(step -> step.outputBuffer() == buffer)) {
                materialized.add(buffer);
            }
        }
        List<LogicalBuffer> elidedInOrder = new ArrayList<>();
        for (LogicalBuffer buffer : temporaries) {
            if (elided.contains(buffer)) {
                elidedInOrder.add(buffer);
            }
        }

        List<CpuBufferBinding> bindings = new ArrayList<>();
        for (LogicalBuffer buffer : schedule.program().buffers()) {
            if (buffer.isExternalInput()) {
                Matrix matrix = mapping == null ? null : mapping.matrixFor(buffer);
                bindings.add(new CpuBufferBinding(buffer, matrix));
            }
        }
        FusionMetrics metrics = FusionMetrics.create(
            strategy,
            schedule.program(),
            steps,
            acceptedRegions,
            elidedInOrder,
            generalized == null ? 0 : generalized.candidateRegionCount(),
            generalized == null ? 0 : generalized.rejectedRegionCount(),
            fusionDecisions);
        return new CpuExecutionPlan(
            schedule, steps, bindings, temporaries, materialized, elidedInOrder, metrics);
    }

    /** Alias with the schedule first, useful when a caller has a transformed schedule. */
    public static CpuExecutionPlan lower(ExecutionPlan executionPlan,
                                         SchedulePlan schedule) {
        return lower(schedule, executionPlan);
    }

    /** Alias with an explicit fusion implementation. */
    public static CpuExecutionPlan lower(ExecutionPlan executionPlan,
                                         SchedulePlan schedule,
                                         FusionStrategy strategy) {
        return lower(schedule, executionPlan, strategy);
    }

    /**
     * Build the bounded default M4 schedule. It starts with M3's initial
     * schedule and applies only the validated scale-then-add fusion family.
     */
    public static SchedulePlan defaultSchedule(AffineProgram program) {
        return defaultSchedule(program, FusionStrategy.fromSystemProperty());
    }

    /** Build the default schedule for an explicit fusion implementation. */
    public static SchedulePlan defaultSchedule(AffineProgram program,
                                               FusionStrategy strategy) {
        if (program == null) {
            throw new IllegalArgumentException("Affine program must not be null");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Fusion strategy must not be null");
        }
        SchedulePlan schedule = SchedulePlan.initial(program);
        if (strategy != FusionStrategy.LEGACY) {
            return schedule;
        }
        int index = 0;
        while (index + 1 < schedule.regions().size()) {
            ScheduleRegion first = schedule.region(index);
            ScheduleRegion second = schedule.region(index + 1);
            if (isSafeScaleAddPair(program, first.statements().get(0), second.statements().get(0))) {
                ScheduleTransformResult result = schedule.fusion(
                    first.primaryStatementId(), second.primaryStatementId());
                if (result.status() == LegalityStatus.LEGAL) {
                    schedule = result.schedule();
                    index++;
                    continue;
                }
            }
            index++;
        }
        return schedule;
    }

    private static CpuStep simpleStep(int id,
                                      ScheduleRegion region,
                                      AffineStatement statement,
                                      IdentityHashMap<AffineStatement, PlanNode> owners) {
        AffineAccess output = firstWrite(statement);
        List<AffineAccess> reads = statement.reads();
        return switch (statement.kind()) {
            case ADD -> {
                requireReadCount(statement, reads, 2);
                yield CpuElementwiseStep.add(
                    id, statement.id(), region.band(), output.buffer(),
                    reads.get(0).buffer(), reads.get(1).buffer(),
                    output, reads.get(0), reads.get(1));
            }
            case SCALE -> {
                requireReadCount(statement, reads, 1);
                yield CpuElementwiseStep.scale(
                    id, statement.id(), region.band(), output.buffer(), reads.get(0).buffer(),
                    scaleFactor(statement, owners), output, reads.get(0));
            }
            case TRANSPOSE -> {
                requireReadCount(statement, reads, 1);
                yield new CpuTransposeStep(
                    id, statement.id(), region.band(), output.buffer(), reads.get(0).buffer(),
                    output, reads.get(0));
            }
            default -> throw new IllegalArgumentException(
                "Unsupported explicit CPU statement kind: " + statement.kind());
        };
    }

    private static CpuGemmStep gemmStep(int id,
                                        AffineStatement init,
                                        AffineStatement update) {
        if (init.kind() != StatementKind.MATMUL_INIT
            || update.kind() != StatementKind.MATMUL_UPDATE) {
            throw new IllegalArgumentException("A GEMM step requires init and update statements");
        }
        AffineAccess output = firstWrite(init);
        List<AffineAccess> reads = update.accesses().stream()
            .filter(access -> access.kind() == AccessKind.READ)
            .toList();
        requireReadCount(update, reads, 2);
        if (update.accesses().stream().noneMatch(
                access -> access.kind() == AccessKind.REDUCTION
                    && access.buffer() == output.buffer())) {
            throw new IllegalArgumentException(
                "MATMUL_UPDATE S" + update.id() + " has no reduction access to its output");
        }
        return new CpuGemmStep(
            id, output.buffer(), reads.get(0).buffer(), reads.get(1).buffer(),
            init.id(), update.id());
    }

    private static FusedSpec fusableRegion(ScheduleRegion region,
                                            AffineProgram program,
                                            IdentityHashMap<AffineStatement, PlanNode> owners) {
        if (region.statements().size() != 2) {
            return null;
        }
        AffineStatement scale = region.statements().get(0);
        AffineStatement add = region.statements().get(1);
        if (!isSafeScaleAddPair(program, scale, add)) {
            return null;
        }

        AffineAccess scaleOutput = firstWrite(scale);
        List<AffineAccess> scaleReads = scale.reads();
        List<AffineAccess> addReads = add.reads();
        AffineAccess fusedAddAccess = null;
        AffineAccess otherAddAccess = null;
        boolean scaledOperandFirst = false;
        for (AffineAccess access : addReads) {
            if (access.buffer() == scaleOutput.buffer()) {
                if (fusedAddAccess != null) {
                    return null;
                }
                fusedAddAccess = access;
                scaledOperandFirst = addReads.get(0) == access;
            } else {
                if (otherAddAccess != null) {
                    return null;
                }
                otherAddAccess = access;
            }
        }
        if (scaleReads.size() != 1 || fusedAddAccess == null || otherAddAccess == null) {
            return null;
        }
        return new FusedSpec(
            scale,
            add,
            region,
            firstWrite(add),
            scaleReads.get(0),
            otherAddAccess,
            scaleOutput.buffer(),
            scaleFactor(scale, owners),
            scaledOperandFirst);
    }

    private static boolean isSafeScaleAddPair(AffineProgram program,
                                               AffineStatement scale,
                                               AffineStatement add) {
        if (scale == null || add == null
            || scale.kind() != StatementKind.SCALE
            || add.kind() != StatementKind.ADD
            || scale.domain().ranges().size() != 2
            || !scale.domain().ranges().equals(add.domain().ranges())) {
            return false;
        }
        AffineAccess scaleOutput = firstWrite(scale);
        if (scale.reads().size() != 1 || add.reads().size() != 2) {
            return false;
        }
        long scaleUses = add.reads().stream()
            .filter(access -> access.buffer() == scaleOutput.buffer())
            .count();
        if (scaleUses != 1) {
            return false;
        }
        for (AffineStatement candidate : program.statements()) {
            if (candidate.id() > add.id()
                && candidate.reads().stream().anyMatch(access -> access.buffer() == scaleOutput.buffer())) {
                return false;
            }
        }
        return true;
    }

    private static AffineStatement matMulUpdateFor(AffineStatement init,
                                                    AffineProgram program) {
        int expectedIndex = init.id() + 1;
        if (expectedIndex >= program.statements().size()) {
            throw new IllegalArgumentException(
                "MATMUL_INIT S" + init.id() + " is not followed by its reduction update");
        }
        AffineStatement expected = program.statements().get(expectedIndex);
        if (expected.kind() == StatementKind.MATMUL_UPDATE
            && expected.accesses().stream().anyMatch(
                access -> access.kind() == AccessKind.REDUCTION
                    && access.buffer() == firstWrite(init).buffer())) {
            return expected;
        }
        throw new IllegalArgumentException(
            "MATMUL_INIT S" + init.id() + " is not followed by its matching reduction update");
    }

    private static AffineAccess firstWrite(AffineStatement statement) {
        return statement.accesses().stream()
            .filter(AffineAccess::writes)
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException(
                "Statement S" + statement.id() + " has no write access"));
    }

    private static void requireReadCount(AffineStatement statement,
                                         List<AffineAccess> reads,
                                         int expected) {
        if (reads.size() != expected) {
            throw new IllegalArgumentException(
                "Statement S" + statement.id() + " expected " + expected
                    + " reads, got " + reads.size());
        }
    }

    private static double scaleFactor(AffineStatement statement,
                                      IdentityHashMap<AffineStatement, PlanNode> owners) {
        PlanNode owner = owners.get(statement);
        if (owner instanceof PlanScale scale) {
            return scale.factor();
        }
        String computation = statement.computation();
        int equals = computation.indexOf('=');
        int multiply = computation.indexOf('*', equals + 1);
        if (equals >= 0 && multiply > equals) {
            try {
                return Double.parseDouble(computation.substring(equals + 1, multiply).trim());
            } catch (NumberFormatException ignored) {
                // The bounded fallback below provides a deterministic diagnostic.
            }
        }
        throw new IllegalArgumentException(
            "Cannot recover SCALE factor for affine statement S" + statement.id());
    }

    private static List<LogicalBuffer> buffersOfKind(AffineProgram program,
                                                     BufferKind kind) {
        return program.buffers().stream().filter(buffer -> buffer.kind() == kind).toList();
    }

    private static final class FusedSpec {
        private final AffineStatement scaleStatement;
        private final AffineStatement addStatement;
        private final ScheduleRegion region;
        private final AffineAccess outputAccess;
        private final AffineAccess scaleAccess;
        private final AffineAccess addAccess;
        private final LogicalBuffer eliminatedBuffer;
        private final double factor;
        private final boolean scaledOperandFirst;

        private FusedSpec(AffineStatement scaleStatement,
                          AffineStatement addStatement,
                          ScheduleRegion region,
                          AffineAccess outputAccess,
                          AffineAccess scaleAccess,
                          AffineAccess addAccess,
                          LogicalBuffer eliminatedBuffer,
                          double factor,
                          boolean scaledOperandFirst) {
            this.scaleStatement = scaleStatement;
            this.addStatement = addStatement;
            this.region = region;
            this.outputAccess = outputAccess;
            this.scaleAccess = scaleAccess;
            this.addAccess = addAccess;
            this.eliminatedBuffer = eliminatedBuffer;
            this.factor = factor;
            this.scaledOperandFirst = scaledOperandFirst;
        }

        private CpuFusedElementwiseStep step(int id) {
            return new CpuFusedElementwiseStep(
                id, scaleStatement.id(), addStatement.id(), region.band(),
                outputAccess.buffer(), scaleAccess.buffer(), addAccess.buffer(),
                eliminatedBuffer, factor, scaledOperandFirst,
                outputAccess, scaleAccess, addAccess);
        }
    }

    /** Mirrors AffineLowerer's deterministic node walk to retain runtime input bindings. */
    private static final class PlanMapping {
        private final AffineProgram program;
        private final IdentityHashMap<PlanNode, LogicalBuffer> nodeBuffers = new IdentityHashMap<>();
        private final IdentityHashMap<AffineStatement, PlanNode> statementOwners
            = new IdentityHashMap<>();
        private final IdentityHashMap<LogicalBuffer, Matrix> matrices = new IdentityHashMap<>();
        private final IdentityHashMap<Matrix, LogicalBuffer> externalBuffers = new IdentityHashMap<>();
        private final IdentityHashMap<MatrixExpr, LogicalBuffer> symbolicBuffers = new IdentityHashMap<>();
        private int bufferCursor;
        private int statementCursor;

        private PlanMapping(ExecutionPlan plan, AffineProgram program) {
            this.program = program;
            visit(plan.root());
            if (bufferCursor != program.buffers().size()
                || statementCursor != program.statements().size()) {
                throw new IllegalArgumentException(
                    "ExecutionPlan does not match the supplied AffineProgram lowering");
            }
        }

        private static PlanMapping create(ExecutionPlan plan, AffineProgram program) {
            return new PlanMapping(plan, program);
        }

        private Matrix matrixFor(LogicalBuffer buffer) {
            return matrices.get(buffer);
        }

        private NodeInfo visit(PlanNode node) {
            LogicalBuffer cached = nodeBuffers.get(node);
            if (cached != null) {
                return new NodeInfo(node, cached);
            }

            LogicalBuffer buffer;
            if (node instanceof PlanInput input) {
                buffer = inputBuffer(input);
            } else if (node instanceof PlanAdd add) {
                visit(add.lhs());
                visit(add.rhs());
                buffer = consumeBuffer(BufferKind.TEMPORARY);
                statementOwners.put(consumeStatement(StatementKind.ADD), node);
            } else if (node instanceof PlanScale scale) {
                visit(scale.operand());
                buffer = consumeBuffer(BufferKind.TEMPORARY);
                statementOwners.put(consumeStatement(StatementKind.SCALE), node);
            } else if (node instanceof PlanTranspose transpose) {
                visit(transpose.operand());
                buffer = consumeBuffer(BufferKind.TEMPORARY);
                statementOwners.put(consumeStatement(StatementKind.TRANSPOSE), node);
            } else if (node instanceof PlanMatMul matMul) {
                visit(matMul.lhs());
                visit(matMul.rhs());
                buffer = consumeBuffer(BufferKind.TEMPORARY);
                statementOwners.put(consumeStatement(StatementKind.MATMUL_INIT), node);
                statementOwners.put(consumeStatement(StatementKind.MATMUL_UPDATE), node);
            } else {
                throw new IllegalArgumentException(
                    "Unsupported execution-plan node: " + node.getClass().getName());
            }
            nodeBuffers.put(node, buffer);
            return new NodeInfo(node, buffer);
        }

        private LogicalBuffer inputBuffer(PlanInput input) {
            MatrixExpr source = input.source();
            if (source instanceof net.faulj.compiler.matrix.Input runtime) {
                Matrix matrix = runtime.matrix();
                LogicalBuffer cached = externalBuffers.get(matrix);
                if (cached != null) {
                    matrices.put(cached, matrix);
                    return cached;
                }
                LogicalBuffer created = consumeBuffer(BufferKind.EXTERNAL_INPUT);
                externalBuffers.put(matrix, created);
                matrices.put(created, matrix);
                return created;
            }
            LogicalBuffer cached = symbolicBuffers.get(source);
            if (cached != null) {
                return cached;
            }
            LogicalBuffer created = consumeBuffer(BufferKind.SYMBOLIC);
            symbolicBuffers.put(source, created);
            return created;
        }

        private LogicalBuffer consumeBuffer(BufferKind expected) {
            if (bufferCursor >= program.buffers().size()) {
                throw new IllegalArgumentException("Affine buffer mapping exhausted");
            }
            LogicalBuffer buffer = program.buffers().get(bufferCursor++);
            if (buffer.kind() != expected) {
                throw new IllegalArgumentException(
                    "Expected " + expected + " buffer, found " + buffer.kind());
            }
            return buffer;
        }

        private AffineStatement consumeStatement(StatementKind expected) {
            if (statementCursor >= program.statements().size()) {
                throw new IllegalArgumentException("Affine statement mapping exhausted");
            }
            AffineStatement statement = program.statements().get(statementCursor++);
            if (statement.kind() != expected) {
                throw new IllegalArgumentException(
                    "Expected " + expected + " statement, found " + statement.kind());
            }
            return statement;
        }
    }

    private record NodeInfo(PlanNode node, LogicalBuffer buffer) {
    }
}
