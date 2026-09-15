package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.Dependence;
import net.faulj.compiler.matrix.affine.DependenceKind;

/**
 * Immutable schedule plan over one M2 {@link AffineProgram}.
 *
 * <p>The affine computation and its dependence graph are retained by
 * reference and never mutated. Every accepted transformation creates a new
 * schedule tree and appends one deterministic history record.</p>
 */
public final class SchedulePlan {
    private final AffineProgram program;
    private final List<ScheduleRegion> regions;
    private final List<ScheduleRegion> initialRegions;
    private final ScheduleSequence root;
    private final ScheduleSequence initialRoot;
    private final List<ScheduleTransformRecord> history;

    private SchedulePlan(AffineProgram program,
                         List<ScheduleRegion> regions,
                         List<ScheduleRegion> initialRegions,
                         List<ScheduleTransformRecord> history) {
        this.program = Objects.requireNonNull(program, "Affine program must not be null");
        this.regions = immutableRegions(regions);
        this.initialRegions = immutableRegions(initialRegions == null ? regions : initialRegions);
        this.history = immutableHistory(history);
        validateRegions(program, this.regions);
        validateRegions(program, this.initialRegions);
        this.root = rootFor(this.regions);
        this.initialRoot = rootFor(this.initialRegions);
    }

    /**
     * Build the deterministic original schedule: one region per statement,
     * with loop order copied from the statement's iteration domain.
     */
    public static SchedulePlan initial(AffineProgram program) {
        Objects.requireNonNull(program, "Affine program must not be null");
        List<ScheduleRegion> regions = new ArrayList<>();
        for (AffineStatement statement : program.statements()) {
            List<ScheduleLoop> loops = statement.domain().ranges().stream()
                .map(range -> new ScheduleLoop(
                    range.variable(),
                    range.lowerInclusive(),
                    range.upperExclusive(),
                    1L))
                .toList();
            regions.add(new ScheduleRegion(
                List.of(statement),
                new ScheduleBand(loops, new ScheduleStatement(statement))));
        }
        return new SchedulePlan(program, regions, regions, List.of());
    }

    public static SchedulePlan from(AffineProgram program) {
        return initial(program);
    }

    /**
     * Construct a schedule from explicit regions. This is useful for
     * inspecting a hand-assembled legal-analysis case; it still validates
     * that every program statement occurs exactly once.
     */
    public static SchedulePlan of(AffineProgram program,
                                  List<ScheduleRegion> regions) {
        return new SchedulePlan(program, regions, regions, List.of());
    }

    public AffineProgram program() {
        return program;
    }

    public AffineProgram affineProgram() {
        return program;
    }

    public ScheduleSequence root() {
        return root;
    }

    public ScheduleSequence schedule() {
        return root;
    }

    public ScheduleSequence scheduleTree() {
        return root;
    }

    public ScheduleSequence initialRoot() {
        return initialRoot;
    }

    public ScheduleSequence initialSchedule() {
        return initialRoot;
    }

    public List<ScheduleRegion> regions() {
        return regions;
    }

    public List<ScheduleRegion> scheduleRegions() {
        return regions;
    }

    public List<ScheduleTransformRecord> history() {
        return history;
    }

    public List<ScheduleTransformRecord> transformationHistory() {
        return history;
    }

    public List<ScheduleTransformRecord> transformations() {
        return history;
    }

    public net.faulj.compiler.matrix.affine.DependenceGraph dependenceGraph() {
        return program.dependenceGraph();
    }

    public boolean hasReductionReassociationMetadata() {
        return regions.stream()
            .flatMap(region -> region.band().loops().stream())
            .anyMatch(loop -> loop.hasAnnotation(ScheduleAnnotation.REDUCTION_REASSOCIATION));
    }

    public boolean hasReductionParallelEligibilityMetadata() {
        return regions.stream()
            .flatMap(region -> region.band().loops().stream())
            .anyMatch(loop -> loop.hasAnnotation(ScheduleAnnotation.REDUCTION_PARALLEL_ELIGIBLE));
    }

    public ScheduleRegion region(int index) {
        return regions.get(index);
    }

    public ScheduleBand scheduleFor(int statementId) {
        ScheduleRegion region = regionForStatement(statementId);
        if (region == null) {
            throw new IllegalArgumentException("Statement S" + statementId + " is not scheduled");
        }
        return region.band();
    }

    public ScheduleTransformResult interchange(int statementId,
                                               AffineVariable first,
                                               AffineVariable second) {
        String operation = "interchange(" + first + "," + second + ") in S" + statementId;
        LegalityResult legality = ScheduleLegality.checkInterchange(this, statementId, first, second);
        if (!legality.isLegal()) {
            return rejected(ScheduleTransformKind.INTERCHANGE, operation, legality);
        }
        ScheduleRegion region = regionForStatement(statementId);
        int firstIndex = region.band().indexOfInductionVariable(first);
        int secondIndex = region.band().indexOfInductionVariable(second);
        int leftIndex = Math.min(firstIndex, secondIndex);
        int rightIndex = Math.max(firstIndex, secondIndex);
        List<ScheduleLoop> loops = new ArrayList<>(region.band().loops());
        ScheduleLoop left = loops.get(leftIndex);
        ScheduleLoop right = loops.get(rightIndex);
        loops.set(leftIndex, right);
        loops.set(rightIndex, left);
        if (program.semantics() != OptimizationSemantics.STRICT
            && hasReassociationReduction(region, left, right)) {
            for (int index = 0; index < loops.size(); index++) {
                ScheduleLoop loop = loops.get(index);
                if (loop.semanticVariable().equals(reductionCarriedVariable(region, left, right))) {
                    loops.set(index, loop.withAnnotation(ScheduleAnnotation.REDUCTION_REASSOCIATION));
                }
            }
        }
        ScheduleRegion updated = region.withBand(region.band().withLoops(loops));
        return accepted(
            ScheduleTransformKind.INTERCHANGE,
            operation,
            legality,
            replaceRegion(region, updated));
    }

    public ScheduleTransformResult interchange(int statementId,
                                               String first,
                                               String second) {
        return interchange(statementId, variable(first), variable(second));
    }

    public ScheduleTransformResult interchange(AffineVariable first,
                                               AffineVariable second) {
        if (regions.size() != 1) {
            return rejected(
                ScheduleTransformKind.INTERCHANGE,
                "interchange(" + first + "," + second + ")",
                LegalityResult.illegal("a statement ID is required when the schedule has multiple regions"));
        }
        return interchange(regions.get(0).primaryStatementId(), first, second);
    }

    public ScheduleTransformResult stripMine(int statementId,
                                             AffineVariable variable,
                                             long tileSize) {
        String operation = "strip-mine(" + variable + "," + tileSize + ") in S" + statementId;
        LegalityResult legality = ScheduleLegality.checkStripMine(this, statementId, variable, tileSize);
        if (!legality.isLegal()) {
            return rejected(ScheduleTransformKind.STRIP_MINE, operation, legality);
        }
        ScheduleRegion region = regionForStatement(statementId);
        int loopIndex = region.band().indexOfInductionVariable(variable);
        ScheduleLoop original = region.band().loop(loopIndex);
        AffineVariable outer = freshVariable(
            original.semanticVariable().name(), "o", region.band().loops(), variable);
        AffineVariable inner = freshVariable(
            original.semanticVariable().name(), "i", region.band().loops(), outer);
        long extent = original.upperBound() - original.lowerBound();
        long outerUpper = ceilDiv(extent, tileSize);
        AffineExpr binding = AffineExpr.variable(outer)
            .scale(tileSize)
            .add(AffineExpr.variable(inner))
            .add(original.lowerBound());
        ScheduleLoop outerLoop = new ScheduleLoop(
            outer,
            original.semanticVariable(),
            0L,
            outerUpper,
            1L,
            List.of(),
            null,
            null,
            "tile " + original.semanticVariable() + " outer");
        ScheduleLoop innerLoop = new ScheduleLoop(
            inner,
            original.semanticVariable(),
            0L,
            tileSize,
            1L,
            original.annotations(),
            binding,
            original.semanticVariable() + " < " + original.upperBound(),
            "tile " + original.semanticVariable() + " inner");
        List<ScheduleLoop> loops = new ArrayList<>(region.band().loops());
        loops.remove(loopIndex);
        loops.add(loopIndex, innerLoop);
        loops.add(loopIndex, outerLoop);
        ScheduleRegion updated = region.withBand(region.band().withLoops(loops));
        return accepted(
            ScheduleTransformKind.STRIP_MINE,
            operation,
            legality,
            replaceRegion(region, updated));
    }

    public ScheduleTransformResult stripMine(int statementId,
                                             String variable,
                                             long tileSize) {
        return stripMine(statementId, variable(variable), tileSize);
    }

    public ScheduleTransformResult tile(int statementId,
                                        AffineVariable variable,
                                        long tileSize) {
        return stripMine(statementId, variable, tileSize);
    }

    public ScheduleTransformResult tile(int statementId,
                                        String variable,
                                        long tileSize) {
        return stripMine(statementId, variable, tileSize);
    }

    public ScheduleTransformResult fuse(int firstStatementId,
                                        int secondStatementId) {
        return fusion(firstStatementId, secondStatementId);
    }

    public ScheduleTransformResult fusion(int firstStatementId,
                                          int secondStatementId) {
        String operation = "fusion(S" + firstStatementId + ",S" + secondStatementId + ")";
        LegalityResult legality = ScheduleLegality.checkFusion(this, firstStatementId, secondStatementId);
        if (!legality.isLegal()) {
            return rejected(ScheduleTransformKind.FUSION, operation, legality);
        }
        ScheduleRegion first = regionForStatement(firstStatementId);
        ScheduleRegion second = regionForStatement(secondStatementId);
        List<ScheduleNode> bodies = List.of(first.band().body(), second.band().body());
        ScheduleBand mergedBand = first.band().withBody(new ScheduleSequence(bodies));
        ScheduleRegion merged = new ScheduleRegion(
            List.of(first.statements().get(0), second.statements().get(0)),
            mergedBand);
        List<ScheduleRegion> updated = new ArrayList<>(regions);
        int firstIndex = indexOfRegion(first);
        updated.set(firstIndex, merged);
        updated.remove(firstIndex + 1);
        return accepted(ScheduleTransformKind.FUSION, operation, legality, updated);
    }

    public ScheduleTransformResult markParallel(int statementId,
                                                AffineVariable variable) {
        return parallel(statementId, variable);
    }

    public ScheduleTransformResult parallel(int statementId,
                                            AffineVariable variable) {
        return markLoop(statementId, variable, ScheduleTransformKind.PARALLEL);
    }

    public ScheduleTransformResult markParallel(int statementId,
                                                String variable) {
        return parallel(statementId, variable);
    }

    public ScheduleTransformResult parallel(int statementId,
                                            String variable) {
        return parallel(statementId, variable(variable));
    }

    public ScheduleTransformResult markVector(int statementId,
                                              AffineVariable variable) {
        return vector(statementId, variable);
    }

    public ScheduleTransformResult vector(int statementId,
                                          AffineVariable variable) {
        return markLoop(statementId, variable, ScheduleTransformKind.VECTOR);
    }

    public ScheduleTransformResult markVector(int statementId,
                                              String variable) {
        return vector(statementId, variable);
    }

    public ScheduleTransformResult vector(int statementId,
                                          String variable) {
        return vector(statementId, variable(variable));
    }

    public ScheduleCandidates candidates() {
        return ScheduleCandidateGenerator.generate(this);
    }

    public ScheduleCandidates generateCandidates() {
        return candidates();
    }

    /**
     * Render the initial schedule, current schedule, and accepted history in
     * deterministic order.
     */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("initial schedule:\n");
        appendRegions(result, initialRegions);
        result.append("schedule:\n");
        appendRegions(result, regions);
        result.append("transformation history:\n");
        if (history.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (ScheduleTransformRecord record : history) {
                result.append("  ").append(record).append('\n');
            }
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    ScheduleRegion regionForStatement(int statementId) {
        for (ScheduleRegion region : regions) {
            if (region.containsStatement(statementId)) {
                return region;
            }
        }
        return null;
    }

    int indexOfRegion(ScheduleRegion target) {
        for (int index = 0; index < regions.size(); index++) {
            if (regions.get(index) == target) {
                return index;
            }
        }
        return -1;
    }

    private ScheduleTransformResult markLoop(int statementId,
                                             AffineVariable variable,
                                             ScheduleTransformKind kind) {
        String operation = kind.displayName() + "(" + variable + ") in S" + statementId;
        LegalityResult legality = kind == ScheduleTransformKind.PARALLEL
            ? ScheduleLegality.checkParallel(this, statementId, variable)
            : ScheduleLegality.checkVector(this, statementId, variable);
        if (!legality.isLegal()) {
            return rejected(kind, operation, legality);
        }
        ScheduleRegion region = regionForStatement(statementId);
        int index = region.band().indexOfInductionVariable(variable);
        ScheduleLoop loop = region.band().loop(index);
        boolean reduction = hasReductionCarriedBy(region, loop.semanticVariable());
        ScheduleAnnotation annotation = kind == ScheduleTransformKind.PARALLEL
            ? ScheduleAnnotation.PARALLEL : ScheduleAnnotation.VECTOR;
        ScheduleLoop updatedLoop;
        if (reduction && kind == ScheduleTransformKind.PARALLEL) {
            // A relaxed reduction is eligible for a future parallel reduction,
            // but M3 records eligibility rather than claiming an executable
            // parallel loop.
            updatedLoop = loop
                .withAnnotation(ScheduleAnnotation.REDUCTION_PARALLEL_ELIGIBLE)
                .withAnnotation(ScheduleAnnotation.REDUCTION_REASSOCIATION);
        } else {
            updatedLoop = loop.withAnnotation(annotation);
            if (reduction) {
                updatedLoop = updatedLoop.withAnnotation(ScheduleAnnotation.REDUCTION_REASSOCIATION);
            }
        }
        List<ScheduleLoop> loops = new ArrayList<>(region.band().loops());
        loops.set(index, updatedLoop);
        ScheduleRegion updated = region.withBand(region.band().withLoops(loops));
        return accepted(kind, operation, legality, replaceRegion(region, updated));
    }

    private ScheduleTransformResult accepted(ScheduleTransformKind kind,
                                              String operation,
                                              LegalityResult legality,
                                              List<ScheduleRegion> updatedRegions) {
        ScheduleTransformRecord record = new ScheduleTransformRecord(
            kind, operation, legality.status(), legality.explanation());
        List<ScheduleTransformRecord> updatedHistory = new ArrayList<>(history);
        updatedHistory.add(record);
        SchedulePlan updated = new SchedulePlan(program, updatedRegions, initialRegions, updatedHistory);
        return new ScheduleTransformResult(this, updated, legality, record);
    }

    private ScheduleTransformResult rejected(ScheduleTransformKind kind,
                                              String operation,
                                              LegalityResult legality) {
        ScheduleTransformRecord record = new ScheduleTransformRecord(
            kind, operation, legality.status(), legality.explanation());
        return new ScheduleTransformResult(this, this, legality, record);
    }

    private List<ScheduleRegion> replaceRegion(ScheduleRegion original,
                                                ScheduleRegion updated) {
        List<ScheduleRegion> result = new ArrayList<>(regions);
        int index = indexOfRegion(original);
        if (index < 0) {
            throw new IllegalStateException("Schedule region disappeared during transformation");
        }
        result.set(index, updated);
        return result;
    }

    private boolean hasReassociationReduction(ScheduleRegion region,
                                               ScheduleLoop first,
                                               ScheduleLoop second) {
        return hasReductionCarriedBy(region, first.semanticVariable())
            || hasReductionCarriedBy(region, second.semanticVariable());
    }

    private AffineVariable reductionCarriedVariable(ScheduleRegion region,
                                                   ScheduleLoop first,
                                                   ScheduleLoop second) {
        for (Dependence dependence : program.dependenceGraph().dependences()) {
            if (!ScheduleLegality.within(dependence, region)
                || dependence.kind() != DependenceKind.REDUCTION) {
                continue;
            }
            Optional<AffineVariable> carried = DependenceDirections.analyze(dependence).carriedBy();
            if (carried.isPresent()
                && (first.semanticVariable().equals(carried.get())
                    || second.semanticVariable().equals(carried.get()))) {
                return carried.get();
            }
        }
        return first.semanticVariable();
    }

    private boolean hasReductionCarriedBy(ScheduleRegion region,
                                          AffineVariable variable) {
        if (program.semantics() == OptimizationSemantics.STRICT) {
            return false;
        }
        for (Dependence dependence : program.dependenceGraph().dependences()) {
            if (ScheduleLegality.within(dependence, region)
                && dependence.kind() == DependenceKind.REDUCTION) {
                Optional<AffineVariable> carried = DependenceDirections.analyze(dependence).carriedBy();
                if (carried.isPresent() && carried.get().equals(variable)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static AffineVariable freshVariable(String base,
                                                String suffix,
                                                List<ScheduleLoop> loops,
                                                AffineVariable extra) {
        String candidate = base + suffix;
        int ordinal = 2;
        while (containsVariable(loops, candidate)
            || (extra != null && extra.name().equals(candidate))) {
            candidate = base + suffix + ordinal++;
        }
        return new AffineVariable(candidate);
    }

    private static boolean containsVariable(List<ScheduleLoop> loops, String name) {
        return loops.stream().anyMatch(loop -> loop.inductionVariable().name().equals(name));
    }

    private static long ceilDiv(long value, long divisor) {
        if (value == 0L) {
            return 0L;
        }
        return value / divisor + (value % divisor == 0L ? 0L : 1L);
    }

    private static AffineVariable variable(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Loop variable name must not be blank");
        }
        return new AffineVariable(name);
    }

    private static ScheduleSequence rootFor(List<ScheduleRegion> regions) {
        return new ScheduleSequence(regions.stream()
            .map(ScheduleRegion::band)
            .map(band -> (ScheduleNode) band)
            .toList());
    }

    private static void appendRegions(StringBuilder result,
                                      List<ScheduleRegion> regions) {
        for (ScheduleRegion region : regions) {
            result.append("  region ").append(region.primaryStatementId());
            if (region.statements().size() > 1) {
                result.append(" (");
                for (int index = 0; index < region.statements().size(); index++) {
                    if (index > 0) {
                        result.append(',');
                    }
                    result.append(region.statements().get(index).name());
                }
                result.append(')');
            }
            result.append(':').append('\n');
            result.append(region.band().dump("    "));
        }
    }

    private static List<ScheduleRegion> immutableRegions(List<ScheduleRegion> source) {
        if (source == null) {
            throw new IllegalArgumentException("Schedule regions must not be null");
        }
        List<ScheduleRegion> copy = new ArrayList<>(source);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Schedule regions must not contain nulls");
        }
        return Collections.unmodifiableList(copy);
    }

    private static List<ScheduleTransformRecord> immutableHistory(
        List<ScheduleTransformRecord> source) {
        if (source == null) {
            throw new IllegalArgumentException("Schedule history must not be null");
        }
        List<ScheduleTransformRecord> copy = new ArrayList<>(source);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Schedule history must not contain nulls");
        }
        return Collections.unmodifiableList(copy);
    }

    private static void validateRegions(AffineProgram program,
                                        List<ScheduleRegion> regions) {
        IdentityHashMap<AffineStatement, Boolean> expected = new IdentityHashMap<>();
        for (AffineStatement statement : program.statements()) {
            expected.put(statement, Boolean.FALSE);
        }
        for (ScheduleRegion region : regions) {
            for (AffineStatement statement : region.statements()) {
                if (!expected.containsKey(statement) || expected.get(statement)) {
                    throw new IllegalArgumentException(
                        "Schedule regions must contain each AffineProgram statement exactly once");
                }
                expected.put(statement, Boolean.TRUE);
            }
        }
        if (expected.values().stream().anyMatch(value -> !value)) {
            throw new IllegalArgumentException(
                "Schedule regions must cover every AffineProgram statement");
        }
    }
}
