package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.Dependence;
import net.faulj.compiler.matrix.affine.DependenceKind;
import net.faulj.compiler.matrix.affine.DependenceStatus;

/**
 * Centralized M3 legality engine.
 *
 * <p>Every schedule transformation goes through this class. An unknown M2
 * fact is never accepted as independence: UNKNOWN is returned to the caller,
 * and the transformation remains unapplied.</p>
 */
public final class ScheduleLegality {
    private ScheduleLegality() {
    }

    public static LegalityResult checkInterchange(SchedulePlan schedule,
                                                  int statementId,
                                                  AffineVariable first,
                                                  AffineVariable second) {
        ScheduleRegion region = regionFor(schedule, statementId);
        if (region == null) {
            return LegalityResult.illegal("statement S" + statementId + " is not in the schedule");
        }
        if (first == null || second == null || first.equals(second)) {
            return LegalityResult.illegal("interchange requires two distinct loop variables");
        }
        int firstIndex = region.band().indexOfInductionVariable(first);
        int secondIndex = region.band().indexOfInductionVariable(second);
        if (firstIndex < 0 || secondIndex < 0) {
            return LegalityResult.illegal(
                "interchange loops " + first + " and " + second + " are not both present in region");
        }
        if (Math.abs(firstIndex - secondIndex) != 1) {
            return LegalityResult.illegal(
                "interchange requires adjacent loops, got " + first + " and " + second);
        }
        int leftIndex = Math.min(firstIndex, secondIndex);
        int rightIndex = Math.max(firstIndex, secondIndex);
        return checkInterchange(schedule, region, leftIndex, rightIndex);
    }

    public static LegalityResult interchange(SchedulePlan schedule,
                                             int statementId,
                                             AffineVariable first,
                                             AffineVariable second) {
        return checkInterchange(schedule, statementId, first, second);
    }

    public static LegalityResult checkStripMine(SchedulePlan schedule,
                                                int statementId,
                                                AffineVariable variable,
                                                long tileSize) {
        ScheduleRegion region = regionFor(schedule, statementId);
        if (region == null) {
            return LegalityResult.illegal("statement S" + statementId + " is not in the schedule");
        }
        if (variable == null) {
            return LegalityResult.illegal("strip-mine requires a loop variable");
        }
        if (tileSize <= 0L) {
            return LegalityResult.illegal("tile size must be positive, got " + tileSize);
        }
        int index = region.band().indexOfInductionVariable(variable);
        if (index < 0) {
            return LegalityResult.illegal("strip-mine loop " + variable + " is not present in region");
        }
        ScheduleLoop loop = region.band().loop(index);
        if (!loop.inductionVariable().equals(loop.semanticVariable())
            || loop.valueExpression() == null
            || !loop.valueExpression().equals(
                net.faulj.compiler.matrix.affine.AffineExpr.variable(loop.inductionVariable()))) {
            return LegalityResult.illegal(
                "re-strip-mining generated tile loops is outside the supported binding model");
        }
        if (loop.step() != 1L) {
            return LegalityResult.illegal(
                "strip-mine supports only unit-step canonical loops, got step " + loop.step());
        }
        return LegalityResult.legal(
            "strip-mining preserves the original lexicographic order of " + variable);
    }

    public static LegalityResult stripMine(SchedulePlan schedule,
                                           int statementId,
                                           AffineVariable variable,
                                           long tileSize) {
        return checkStripMine(schedule, statementId, variable, tileSize);
    }

    public static LegalityResult checkFusion(SchedulePlan schedule,
                                             int firstStatementId,
                                             int secondStatementId) {
        if (schedule == null) {
            return LegalityResult.illegal("schedule must not be null");
        }
        ScheduleRegion first = regionFor(schedule, firstStatementId);
        ScheduleRegion second = regionFor(schedule, secondStatementId);
        if (first == null || second == null) {
            return LegalityResult.illegal("fusion statements must both be present in the schedule");
        }
        if (first == second) {
            return LegalityResult.illegal("fusion requires statements from distinct regions");
        }
        int firstIndex = schedule.indexOfRegion(first);
        int secondIndex = schedule.indexOfRegion(second);
        if (secondIndex != firstIndex + 1) {
            return LegalityResult.illegal("fusion requires adjacent schedule regions");
        }
        if (!first.isSingleton() || !second.isSingleton()) {
            return LegalityResult.illegal("fusion currently accepts only singleton statement regions");
        }
        AffineStatement firstStatement = first.statements().get(0);
        AffineStatement secondStatement = second.statements().get(0);
        if (!firstStatement.domain().ranges().equals(secondStatement.domain().ranges())) {
            return LegalityResult.illegal(
                "incompatible iteration domains for " + firstStatement.name() + " and "
                    + secondStatement.name());
        }
        if (!compatibleBands(first.band(), second.band())) {
            return LegalityResult.illegal(
                "incompatible loop bands for " + firstStatement.name() + " and "
                    + secondStatement.name());
        }

        LegalityResult availability = checkFusedAvailability(firstStatement, secondStatement);
        if (!availability.isLegal()) {
            return availability;
        }

        for (Dependence dependence : schedule.dependenceGraph().dependences()) {
            boolean firstToSecond = dependence.source() == firstStatement
                && dependence.sink() == secondStatement;
            boolean secondToFirst = dependence.source() == secondStatement
                && dependence.sink() == firstStatement;
            if (!firstToSecond && !secondToFirst) {
                continue;
            }
            if (dependence.status() == DependenceStatus.UNKNOWN) {
                return LegalityResult.unknown(
                    "possible alias/dependence between " + dependence.source().name()
                        + " and " + dependence.sink().name() + " is unknown");
            }
            if (secondToFirst) {
                return LegalityResult.illegal(
                    dependence.source().name() + " -> " + dependence.sink().name()
                        + " would be reversed by fusion");
            }
        }
        return LegalityResult.legal(
            "identical domains and loop bands; statement order and all proven dependences are preserved");
    }

    public static LegalityResult fusion(SchedulePlan schedule,
                                        int firstStatementId,
                                        int secondStatementId) {
        return checkFusion(schedule, firstStatementId, secondStatementId);
    }

    public static LegalityResult checkParallel(SchedulePlan schedule,
                                               int statementId,
                                               AffineVariable variable) {
        return checkLoopMark(schedule, statementId, variable, ScheduleTransformKind.PARALLEL);
    }

    public static LegalityResult parallel(SchedulePlan schedule,
                                          int statementId,
                                          AffineVariable variable) {
        return checkParallel(schedule, statementId, variable);
    }

    public static LegalityResult checkVector(SchedulePlan schedule,
                                             int statementId,
                                             AffineVariable variable) {
        return checkLoopMark(schedule, statementId, variable, ScheduleTransformKind.VECTOR);
    }

    public static LegalityResult vector(SchedulePlan schedule,
                                        int statementId,
                                        AffineVariable variable) {
        return checkVector(schedule, statementId, variable);
    }

    static LegalityResult checkInterchange(SchedulePlan schedule,
                                           ScheduleRegion region,
                                           int leftIndex,
                                           int rightIndex) {
        List<ScheduleLoop> originalLoops = region.band().loops();
        List<ScheduleLoop> proposedLoops = new ArrayList<>(originalLoops);
        ScheduleLoop left = proposedLoops.get(leftIndex);
        ScheduleLoop right = proposedLoops.get(rightIndex);
        proposedLoops.set(leftIndex, right);
        proposedLoops.set(rightIndex, left);
        if (!bindingsFollowDefinitions(proposedLoops)) {
            return LegalityResult.illegal(
                "interchange would move a binder loop after a derived index that depends on it");
        }
        boolean reassociation = false;

        for (Dependence dependence : schedule.dependenceGraph().dependences()) {
            if (!within(dependence, region)) {
                continue;
            }
            if (dependence.status() == DependenceStatus.UNKNOWN) {
                return LegalityResult.unknown(
                    "dependence " + dependence.source().name() + " -> "
                        + dependence.sink().name() + " has UNKNOWN alias/direction information");
            }
            if (dependence.kind() == DependenceKind.REDUCTION) {
                Optional<AffineVariable> carried = DependenceDirections.analyze(dependence).carriedBy();
                if (carried.isEmpty()) {
                    return LegalityResult.unknown(
                        "reduction dependence " + dependence.source().name()
                            + " -> " + dependence.sink().name()
                            + " has no recognized carried dimension");
                }
                if (usesSemanticVariable(left, carried.get())
                    || usesSemanticVariable(right, carried.get())) {
                    if (schedule.program().semantics() == OptimizationSemantics.STRICT) {
                        return LegalityResult.illegal(
                            "STRICT reduction dependence is carried by " + carried.get()
                                + "; interchange would change reduction ordering");
                    }
                    reassociation = true;
                }
                continue;
            }
            DependenceDirectionInfo directions = DependenceDirections.analyze(dependence);
            for (ScheduleLoop loop : proposedLoops) {
                DependenceDirection direction = directions.direction(loop.semanticVariable());
                if (direction == DependenceDirection.UNKNOWN) {
                    return LegalityResult.unknown(
                        "dependence " + dependence.source().name() + " -> "
                            + dependence.sink().name() + " has UNKNOWN direction for "
                            + loop.semanticVariable());
                }
                if (direction == DependenceDirection.GREATER) {
                    return LegalityResult.illegal(
                        dependence.source().name() + " -> " + dependence.sink().name()
                            + " has a negative direction after interchange at "
                            + loop.semanticVariable());
                }
                if (direction == DependenceDirection.LESS) {
                    break;
                }
            }
        }
        if (reassociation) {
            return LegalityResult.legal(
                "only reassociation-eligible reduction ordering is changed; metadata will record it");
        }
        return LegalityResult.legal("no dependence carried in the proposed loop order");
    }

    static LegalityResult checkLoopMark(SchedulePlan schedule,
                                        int statementId,
                                        AffineVariable variable,
                                        ScheduleTransformKind kind) {
        ScheduleRegion region = regionFor(schedule, statementId);
        if (region == null) {
            return LegalityResult.illegal("statement S" + statementId + " is not in the schedule");
        }
        if (variable == null) {
            return LegalityResult.illegal(kind + " marking requires a loop variable");
        }
        int loopIndex = region.band().indexOfInductionVariable(variable);
        if (loopIndex < 0) {
            return LegalityResult.illegal("loop " + variable + " is not present in region");
        }
        ScheduleLoop loop = region.band().loop(loopIndex);
        boolean reductionRelaxation = false;
        for (Dependence dependence : schedule.dependenceGraph().dependences()) {
            if (!within(dependence, region)) {
                continue;
            }
            if (dependence.status() == DependenceStatus.UNKNOWN) {
                return LegalityResult.unknown(
                    "dependence " + dependence.source().name() + " -> "
                        + dependence.sink().name() + " is UNKNOWN");
            }
            if (dependence.kind() == DependenceKind.REDUCTION) {
                Optional<AffineVariable> carried = DependenceDirections.analyze(dependence).carriedBy();
                if (carried.isEmpty()) {
                    return LegalityResult.unknown(
                        "reduction dependence " + dependence.source().name()
                            + " -> " + dependence.sink().name()
                            + " has unknown carried dimension");
                }
                if (loop.semanticVariable().equals(carried.get())) {
                    if (schedule.program().semantics() == OptimizationSemantics.STRICT) {
                        return LegalityResult.illegal(
                            "STRICT reduction dimension " + variable + " cannot be marked "
                                + kind.displayName());
                    }
                    reductionRelaxation = true;
                }
                continue;
            }
            DependenceDirection direction = DependenceDirections.analyze(dependence)
                .direction(loop.semanticVariable());
            if (direction == DependenceDirection.UNKNOWN) {
                return LegalityResult.unknown(
                    "dependence " + dependence.source().name() + " -> "
                        + dependence.sink().name() + " has UNKNOWN direction for "
                        + loop.semanticVariable());
            }
            if (direction != DependenceDirection.EQUAL) {
                return LegalityResult.illegal(
                    dependence.source().name() + " -> " + dependence.sink().name()
                        + " is carried by " + loop.semanticVariable());
            }
        }
        if (reductionRelaxation) {
            return LegalityResult.legal(
                "reassociation-eligible reduction; " + kind.displayName()
                    + " is recorded as metadata only");
        }
        return LegalityResult.legal("no loop-carried non-reduction dependence prevents "
            + kind.displayName());
    }

    static boolean within(Dependence dependence, ScheduleRegion region) {
        return region.contains(dependence.source()) && region.contains(dependence.sink());
    }

    static boolean compatibleBands(ScheduleBand first, ScheduleBand second) {
        if (first.loops().size() != second.loops().size()) {
            return false;
        }
        for (int index = 0; index < first.loops().size(); index++) {
            ScheduleLoop left = first.loop(index);
            ScheduleLoop right = second.loop(index);
            if (!left.semanticVariable().equals(right.semanticVariable())
                || left.lowerBound() != right.lowerBound()
                || left.upperBound() != right.upperBound()
                || left.step() != right.step()) {
                return false;
            }
        }
        return true;
    }

    private static boolean usesSemanticVariable(ScheduleLoop loop,
                                                AffineVariable variable) {
        return loop.semanticVariable().equals(variable);
    }

    private static LegalityResult checkFusedAvailability(AffineStatement producer,
                                                          AffineStatement consumer) {
        for (AffineAccess first : producer.accesses()) {
            for (AffineAccess second : consumer.accesses()) {
                if (first.buffer() != second.buffer()
                    || (!first.writes() && !second.writes())) {
                    continue;
                }
                // The original two-region order puts every producer point before
                // every consumer point. Interleaving is safe only when a shared
                // location belongs to the same unique logical point in both.
                if (!first.indices().equals(second.indices())
                    || !isInjectiveUnitProjection(first, producer)
                    || !isInjectiveUnitProjection(second, consumer)) {
                    return LegalityResult.unknown(
                        "fusion cannot prove RAW/WAR/WAW ordering across iterations");
                }
            }
        }
        return LegalityResult.legal(
            "all shared RAW/WAR/WAW locations are unique to the same fused iteration");
    }

    private static boolean isInjectiveUnitProjection(AffineAccess access,
                                                      AffineStatement statement) {
        java.util.Set<AffineVariable> seen = new java.util.HashSet<>();
        for (AffineExpr index : access.indices()) {
            if (index.coefficients().size() != 1
                || index.coefficients().firstEntry().getValue() != 1L
                || !seen.add(index.coefficients().firstKey())) {
                return false;
            }
        }
        return seen.containsAll(statement.domain().variables())
            && seen.size() == statement.domain().variables().size();
    }

    private static boolean bindingsFollowDefinitions(List<ScheduleLoop> loops) {
        java.util.Set<AffineVariable> available = new java.util.HashSet<>();
        for (ScheduleLoop loop : loops) {
            available.add(loop.inductionVariable());
            if (loop.valueExpression() != null
                && !available.containsAll(loop.valueExpression().coefficients().keySet())) {
                return false;
            }
            if (loop.valueExpression() != null) {
                available.add(loop.semanticVariable());
            }
        }
        return true;
    }

    private static ScheduleRegion regionFor(SchedulePlan schedule, int statementId) {
        if (schedule == null) {
            return null;
        }
        return schedule.regionForStatement(statementId);
    }
}
