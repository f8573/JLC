package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;

/**
 * Inspectable semantic description of one generalized R2 fused region.
 *
 * <p>A region has one materialized output.  Its leaf buffers may be borrowed
 * inputs or materialized values produced outside the region; every other
 * logical value listed in {@link #eliminatedBuffers()} is absent from the
 * executable step list and therefore cannot receive an R1 physical slot.</p>
 */
public final class FusedRegionPlan {
    private final int regionId;
    private final int startRegionIndex;
    private final int endRegionIndex;
    private final List<ScheduleRegion> scheduleRegions;
    private final List<AffineStatement> statements;
    private final ScheduleBand scheduleBand;
    private final IterationDomain iterationDomain;
    private final LogicalBuffer outputBuffer;
    private final AffineAccess outputAccess;
    private final List<LogicalBuffer> leafBuffers;
    private final List<LogicalBuffer> internalValues;
    private final List<LogicalBuffer> eliminatedBuffers;
    private final ScalarFusionProgram scalarProgram;
    private final LegalityStatus legality;
    private final String legalityExplanation;
    private final String profitability;

    FusedRegionPlan(int regionId,
                    int startRegionIndex,
                    int endRegionIndex,
                    List<ScheduleRegion> scheduleRegions,
                    List<AffineStatement> statements,
                    ScheduleBand scheduleBand,
                    IterationDomain iterationDomain,
                    LogicalBuffer outputBuffer,
                    AffineAccess outputAccess,
                    List<LogicalBuffer> leafBuffers,
                    List<LogicalBuffer> internalValues,
                    List<LogicalBuffer> eliminatedBuffers,
                    ScalarFusionProgram scalarProgram,
                    LegalityStatus legality,
                    String legalityExplanation,
                    String profitability) {
        this.regionId = requireNonNegative(regionId, "Region ID");
        this.startRegionIndex = requireNonNegative(startRegionIndex, "Region start");
        this.endRegionIndex = requireNonNegative(endRegionIndex, "Region end");
        if (endRegionIndex < startRegionIndex) {
            throw new IllegalArgumentException("Fused region end must not precede its start");
        }
        this.scheduleRegions = immutableCopy(scheduleRegions, "Schedule regions");
        this.statements = immutableCopy(statements, "Fused statements");
        this.scheduleBand = Objects.requireNonNull(scheduleBand, "Fused schedule band must not be null");
        this.iterationDomain = Objects.requireNonNull(
            iterationDomain, "Fused iteration domain must not be null");
        this.outputBuffer = Objects.requireNonNull(outputBuffer, "Fused output buffer must not be null");
        this.outputAccess = Objects.requireNonNull(outputAccess, "Fused output access must not be null");
        this.leafBuffers = immutableCopy(leafBuffers, "Fused leaf buffers");
        this.internalValues = immutableCopy(internalValues, "Fused internal values");
        this.eliminatedBuffers = immutableCopy(eliminatedBuffers, "Fused eliminated buffers");
        this.scalarProgram = Objects.requireNonNull(
            scalarProgram, "Fused scalar program must not be null");
        this.legality = Objects.requireNonNull(legality, "Fused legality must not be null");
        if (legality != LegalityStatus.LEGAL) {
            throw new IllegalArgumentException("A realized fused region must be LEGAL");
        }
        this.legalityExplanation = requireText(legalityExplanation, "Legality explanation");
        this.profitability = requireText(profitability, "Profitability explanation");
        if (statements.size() < 2) {
            throw new IllegalArgumentException("A fused region must contain at least two statements");
        }
        if (eliminatedBuffers.isEmpty()) {
            throw new IllegalArgumentException("A fused region must eliminate a materialization");
        }
        if (!outputBuffer.isTemporary()) {
            throw new IllegalArgumentException("A fused output must be an owned temporary");
        }
        validateIdentityMembership();
    }

    public int regionId() {
        return regionId;
    }

    public int startRegionIndex() {
        return startRegionIndex;
    }

    public int endRegionIndex() {
        return endRegionIndex;
    }

    public List<ScheduleRegion> scheduleRegions() {
        return scheduleRegions;
    }

    public List<AffineStatement> statements() {
        return statements;
    }

    public List<Integer> statementIds() {
        return statements.stream().map(AffineStatement::id).toList();
    }

    public ScheduleBand scheduleBand() {
        return scheduleBand;
    }

    public IterationDomain iterationDomain() {
        return iterationDomain;
    }

    public LogicalBuffer outputBuffer() {
        return outputBuffer;
    }

    public AffineAccess outputAccess() {
        return outputAccess;
    }

    public List<LogicalBuffer> leafBuffers() {
        return leafBuffers;
    }

    public List<LogicalBuffer> externalLeafBuffers() {
        return leafBuffers;
    }

    public List<LogicalBuffer> internalValues() {
        return internalValues;
    }

    public List<LogicalBuffer> eliminatedBuffers() {
        return eliminatedBuffers;
    }

    public List<LogicalBuffer> eliminatedTemporaryBuffers() {
        return eliminatedBuffers;
    }

    public ScalarFusionProgram scalarProgram() {
        return scalarProgram;
    }

    public LegalityStatus legality() {
        return legality;
    }

    public String legalityExplanation() {
        return legalityExplanation;
    }

    public String profitability() {
        return profitability;
    }

    CpuFusedRegionStep step(int stepId) {
        return new CpuFusedRegionStep(stepId, this);
    }

    /** Deterministic region dump used by {@link CpuExecutionPlan}. */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("region F").append(regionId).append(':').append('\n');
        result.append("  statements=").append(statementIds()).append('\n');
        result.append("  output=%").append(outputBuffer.id()).append(' ')
            .append(outputBuffer.name()).append('\n');
        result.append("  iteration-space=").append(iterationDomain).append('\n');
        result.append("  leaves=").append(formatBuffers(leafBuffers)).append('\n');
        result.append("  internal=").append(formatBuffers(internalValues)).append('\n');
        result.append(scalarProgram.dump().indent(2));
        result.append("  store %").append(outputBuffer.id()).append(outputAccess.location()
            .substring(outputAccess.location().indexOf('['))).append(" = v")
            .append(scalarProgram.rootNodeId()).append('\n');
        result.append("  eliminated=").append(formatBuffers(eliminatedBuffers)).append('\n');
        result.append("  legality=").append(legality).append(" (").append(legalityExplanation)
            .append(')')
            .append('\n');
        result.append("  profitability=").append(profitability).append('\n');
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private void validateIdentityMembership() {
        Set<AffineStatement> seenStatements = Collections.newSetFromMap(new IdentityHashMap<>());
        for (AffineStatement statement : statements) {
            if (!seenStatements.add(statement)) {
                throw new IllegalArgumentException("Fused statements must be unique");
            }
        }
        if (internalValues.stream().anyMatch(buffer -> !eliminatedBuffers.contains(buffer))) {
            throw new IllegalArgumentException("Every fused internal value must be eliminated");
        }
        if (eliminatedBuffers.stream().anyMatch(buffer -> !buffer.isTemporary())) {
            throw new IllegalArgumentException("Only temporary buffers may be eliminated");
        }
        if (leafBuffers.stream().anyMatch(buffer -> eliminatedBuffers.contains(buffer))) {
            throw new IllegalArgumentException("A buffer cannot be both a leaf and eliminated");
        }
    }

    private static String formatBuffers(List<LogicalBuffer> buffers) {
        List<String> names = new ArrayList<>(buffers.size());
        for (LogicalBuffer buffer : buffers) {
            names.add("%" + buffer.id());
        }
        return names.toString();
    }

    private static int requireNonNegative(int value, String role) {
        if (value < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
        return value;
    }

    private static String requireText(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " must not be blank");
        }
        return value;
    }

    private static <T> List<T> immutableCopy(List<T> source, String role) {
        if (source == null || source.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(role + " must not be null or contain nulls");
        }
        return Collections.unmodifiableList(new ArrayList<>(source));
    }
}
