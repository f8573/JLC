package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import net.faulj.compiler.matrix.affine.AccessKind;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.Dependence;
import net.faulj.compiler.matrix.affine.DependenceStatus;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.LegalityStatus;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;
import net.faulj.compiler.matrix.schedule.SchedulePlan;

/**
 * Deterministic post-M4 planner for generalized elementwise fusion.
 *
 * <p>The planner scans the incoming schedule in execution order, grows a
 * connected region greedily, and then validates the complete candidate.  It
 * does not rewrite the M3 schedule or perform algebraic reassociation.  A
 * failed candidate is left as ordinary materialized CPU steps.</p>
 */
public final class GeneralizedFusionPlanner {
    private GeneralizedFusionPlanner() {
    }

    public static Result plan(SchedulePlan schedule) {
        if (schedule == null) {
            throw new IllegalArgumentException("Schedule plan must not be null");
        }
        return new Builder(schedule).build();
    }

    /** Immutable result consumed by {@link CpuLowerer}. */
    public static final class Result {
        private final IdentityHashMap<ScheduleRegion, FusedRegionPlan> byStart;
        private final List<FusedRegionPlan> acceptedRegions;
        private final List<String> decisions;
        private final int candidateRegionCount;
        private final int rejectedRegionCount;

        private Result(IdentityHashMap<ScheduleRegion, FusedRegionPlan> byStart,
                       List<FusedRegionPlan> acceptedRegions,
                       List<String> decisions,
                       int candidateRegionCount,
                       int rejectedRegionCount) {
            this.byStart = new IdentityHashMap<>(byStart);
            this.acceptedRegions = List.copyOf(acceptedRegions);
            this.decisions = List.copyOf(decisions);
            this.candidateRegionCount = candidateRegionCount;
            this.rejectedRegionCount = rejectedRegionCount;
        }

        public List<FusedRegionPlan> acceptedRegions() {
            return acceptedRegions;
        }

        public List<String> decisions() {
            return decisions;
        }

        public int candidateRegionCount() {
            return candidateRegionCount;
        }

        public int rejectedRegionCount() {
            return rejectedRegionCount;
        }

        public FusedRegionPlan startingAt(ScheduleRegion region) {
            return byStart.get(region);
        }
    }

    private static final class Builder {
        private final SchedulePlan schedule;
        private final List<ScheduleRegion> regions;
        private final IdentityHashMap<ScheduleRegion, FusedRegionPlan> acceptedByStart
            = new IdentityHashMap<>();
        private final List<FusedRegionPlan> acceptedRegions = new ArrayList<>();
        private final List<String> decisions = new ArrayList<>();
        private int candidateRegionCount;
        private int rejectedRegionCount;
        private int nextRegionId;

        private Builder(SchedulePlan schedule) {
            this.schedule = schedule;
            this.regions = schedule.regions();
        }

        private Result build() {
            int start = 0;
            while (start < regions.size()) {
                int maxEnd = growEnd(start);
                int statementCount = countStatements(start, maxEnd);
                if (statementCount < 2) {
                    if (maxEnd == start && start + 1 < regions.size()
                        && regionEligible(regions.get(start))
                        && !regionEligible(regions.get(start + 1))) {
                        decisions.add("fusion boundary after "
                            + regions.get(start).statements().get(0).name() + ": "
                            + boundaryReason(regions.get(start + 1)));
                    }
                    start++;
                    continue;
                }
                candidateRegionCount++;
                FusedRegionPlan accepted = null;
                String lastRejection = null;
                for (int end = maxEnd; end >= start; end--) {
                    if (countStatements(start, end) < 2) {
                        continue;
                    }
                    try {
                        accepted = buildCandidate(start, end);
                        break;
                    } catch (FusionRejected rejection) {
                        lastRejection = rejection.getMessage();
                    }
                }
                if (accepted == null) {
                    rejectedRegionCount++;
                    decisions.add("candidate regions " + start + ".." + maxEnd
                        + " rejected: " + (lastRejection == null ? "not profitable" : lastRejection));
                    start++;
                } else {
                    acceptedByStart.put(regions.get(start), accepted);
                    acceptedRegions.add(accepted);
                    decisions.add("region F" + accepted.regionId() + " accepted: "
                        + accepted.legalityExplanation() + "; " + accepted.profitability());
                    start = accepted.endRegionIndex() + 1;
                }
            }
            return new Result(
                acceptedByStart,
                acceptedRegions,
                decisions,
                candidateRegionCount,
                rejectedRegionCount);
        }

        private int growEnd(int start) {
            if (!regionEligible(regions.get(start))) {
                return start;
            }
            Set<LogicalBuffer> produced = Collections.newSetFromMap(new IdentityHashMap<>());
            addOutputs(regions.get(start), produced);
            int end = start;
            while (end + 1 < regions.size()) {
                ScheduleRegion next = regions.get(end + 1);
                if (!regionEligible(next) || !consumesProduced(next, produced)) {
                    break;
                }
                addOutputs(next, produced);
                end++;
            }
            return end;
        }

        private FusedRegionPlan buildCandidate(int start, int end) {
            List<ScheduleRegion> candidateRegions = new ArrayList<>();
            List<AffineStatement> statements = new ArrayList<>();
            for (int index = start; index <= end; index++) {
                ScheduleRegion region = regions.get(index);
                candidateRegions.add(region);
                statements.addAll(region.statements());
            }
            if (statements.size() < 2) {
                throw reject("candidate has fewer than two statements");
            }
            Set<AffineStatement> inside = Collections.newSetFromMap(new IdentityHashMap<>());
            IdentityHashMap<AffineStatement, Integer> positions = new IdentityHashMap<>();
            for (int index = 0; index < statements.size(); index++) {
                AffineStatement statement = statements.get(index);
                inside.add(statement);
                positions.put(statement, index);
                validateStatementShape(statement);
            }

            AffineStatement finalStatement = statements.get(statements.size() - 1);
            AffineAccess outputAccess = firstWrite(finalStatement);
            LogicalBuffer output = outputAccess.buffer();
            Set<LogicalBuffer> eliminated = Collections.newSetFromMap(new IdentityHashMap<>());
            List<LogicalBuffer> internal = new ArrayList<>();
            for (int index = 0; index < statements.size() - 1; index++) {
                LogicalBuffer buffer = firstWrite(statements.get(index)).buffer();
                if (!buffer.isTemporary()) {
                    throw reject("statement " + statements.get(index).name()
                        + " writes a non-temporary buffer");
                }
                internal.add(buffer);
                eliminated.add(buffer);
            }
            for (LogicalBuffer buffer : internal) {
                for (AffineStatement consumer : schedule.program().statements()) {
                    if (consumer.reads().stream().anyMatch(access -> access.buffer() == buffer)
                        && !inside.contains(consumer)) {
                        throw reject("escaping producer %" + buffer.id()
                            + " is consumed by " + consumer.name());
                    }
                }
            }

            for (Dependence dependence : schedule.dependenceGraph().dependences()) {
                boolean touches = inside.contains(dependence.source())
                    || inside.contains(dependence.sink());
                if (!touches) {
                    continue;
                }
                if (dependence.status() == DependenceStatus.UNKNOWN) {
                    throw reject("UNKNOWN alias/dependence for "
                        + dependence.source().name() + " -> " + dependence.sink().name());
                }
                Integer sourcePosition = positions.get(dependence.source());
                Integer sinkPosition = positions.get(dependence.sink());
                if (sourcePosition != null && sinkPosition != null
                    && sourcePosition > sinkPosition) {
                    throw reject("statement order would reverse "
                        + dependence.source().name() + " -> " + dependence.sink().name());
                }
            }

            if (output == null || !output.isTemporary()) {
                throw reject("fused output is not a materialized temporary");
            }
            ScheduleBand band = candidateRegions.get(candidateRegions.size() - 1).band();
            validateScheduleBindings(finalStatement.domain(), band);
            ScalarBuilder scalarBuilder = new ScalarBuilder(
                statements, finalStatement.domain());
            int root = scalarBuilder.buildValue(
                output, List.copyOf(outputAccess.indices()));
            ScalarFusionProgram scalarProgram = scalarBuilder.finish(root);

            Set<LogicalBuffer> representedProducers =
                Collections.newSetFromMap(new IdentityHashMap<>());
            for (ScalarFusionProgram.ScalarNode node : scalarProgram.nodes()) {
                if (!node.isLoad()) {
                    representedProducers.add(node.buffer());
                }
            }
            if (!representedProducers.containsAll(eliminated)) {
                throw reject("candidate contains an internal producer outside the final scalar path");
            }

            List<LogicalBuffer> leaves = scalarBuilder.leaves();
            String legality = "PROVEN dependence order, pointwise producer-consumer maps, "
                + "and no escaping internal value";
            String profitability = eliminated.size() + " materializations removed; "
                + (statements.size() - 1) + " full-matrix passes collapsed to one"
                + (containsTranspose(statements) ? "; transpose map cost accepted" : "");
            return new FusedRegionPlan(
                nextRegionId++, start, end, candidateRegions, statements, band,
                finalStatement.domain(), output, outputAccess, leaves, internal,
                new ArrayList<>(internal), scalarProgram, LegalityStatus.LEGAL,
                legality, profitability);
        }

        private void validateStatementShape(AffineStatement statement) {
            if (statement.kind() != StatementKind.ADD
                && statement.kind() != StatementKind.SCALE
                && statement.kind() != StatementKind.TRANSPOSE) {
                throw reject(statement.kind() == StatementKind.MATMUL_INIT
                    || statement.kind() == StatementKind.MATMUL_UPDATE
                    ? "GEMM/reduction boundary at " + statement.name()
                    : "unsupported operation " + statement.kind());
            }
            if (statement.hasReduction() || statement.domain().ranges().size() != 2) {
                throw reject("unsupported reduction or iteration domain at " + statement.name());
            }
            List<AffineAccess> writes = statement.writes();
            List<AffineAccess> reads = statement.reads();
            int expectedReads = switch (statement.kind()) {
                case ADD -> 2;
                case SCALE, TRANSPOSE -> 1;
                default -> -1;
            };
            if (writes.size() != 1 || reads.size() != expectedReads) {
                throw reject("unsupported access arity at " + statement.name());
            }
            for (AffineAccess access : statement.accesses()) {
                if (access.kind() == AccessKind.REDUCTION || access.indices().size() != 2) {
                    throw reject("unsupported affine access at " + statement.name());
                }
            }
            if (!writes.get(0).buffer().isTemporary()) {
                throw reject(statement.name() + " writes an external or symbolic buffer");
            }
        }

        private void validateScheduleBindings(IterationDomain domain, ScheduleBand band) {
            for (AffineVariable variable : domain.variables()) {
                boolean bound = band.loops().stream()
                    .anyMatch(loop -> loop.semanticVariable().equals(variable));
                if (!bound) {
                    throw reject("incompatible domain: schedule does not bind " + variable);
                }
            }
        }

        private static void addOutputs(ScheduleRegion region, Set<LogicalBuffer> result) {
            for (AffineStatement statement : region.statements()) {
                for (AffineAccess write : statement.writes()) {
                    result.add(write.buffer());
                }
            }
        }

        private static boolean consumesProduced(ScheduleRegion region,
                                                 Set<LogicalBuffer> produced) {
            for (AffineStatement statement : region.statements()) {
                if (statement.reads().stream().anyMatch(access -> produced.contains(access.buffer()))) {
                    return true;
                }
            }
            return false;
        }

        private static boolean regionEligible(ScheduleRegion region) {
            return region.statements().stream().allMatch(statement ->
                statement.kind() == StatementKind.ADD
                    || statement.kind() == StatementKind.SCALE
                    || statement.kind() == StatementKind.TRANSPOSE);
        }

        private static int countStatements(int start, int end, List<ScheduleRegion> regions) {
            int result = 0;
            for (int index = start; index <= end; index++) {
                result += regions.get(index).statements().size();
            }
            return result;
        }

        private int countStatements(int start, int end) {
            return countStatements(start, end, regions);
        }

        private static AffineAccess firstWrite(AffineStatement statement) {
            return statement.writes().stream().findFirst().orElseThrow(
                () -> reject("statement " + statement.name() + " has no write"));
        }

        private static boolean containsTranspose(List<AffineStatement> statements) {
            return statements.stream().anyMatch(
                statement -> statement.kind() == StatementKind.TRANSPOSE);
        }

        private static String boundaryReason(ScheduleRegion region) {
            if (region.statements().stream().anyMatch(statement ->
                statement.kind() == StatementKind.MATMUL_INIT
                    || statement.kind() == StatementKind.MATMUL_UPDATE)) {
                return "GEMM/reduction boundary at "
                    + region.statements().get(0).name();
            }
            return "unsupported or non-elementwise boundary at "
                + region.statements().get(0).name();
        }
    }

    private static final class ScalarBuilder {
        private final List<AffineStatement> orderedStatements;
        private final IterationDomain canonicalDomain;
        private final IdentityHashMap<LogicalBuffer, AffineStatement> producers
            = new IdentityHashMap<>();
        private final List<ScalarFusionProgram.ScalarNode> nodes = new ArrayList<>();
        private final Map<ValueKey, Integer> cachedValues = new HashMap<>();
        private final List<LogicalBuffer> leaves = new ArrayList<>();
        private final Set<LogicalBuffer> leafSet = Collections.newSetFromMap(new IdentityHashMap<>());
        private final Set<ValueKey> active = new HashSet<>();

        private ScalarBuilder(List<AffineStatement> orderedStatements,
                              IterationDomain canonicalDomain) {
            this.orderedStatements = List.copyOf(orderedStatements);
            this.canonicalDomain = canonicalDomain;
            for (AffineStatement statement : this.orderedStatements) {
                for (AffineAccess write : statement.writes()) {
                    if (producers.put(write.buffer(), statement) != null) {
                        throw reject("multiple scalar producers for %" + write.buffer().id());
                    }
                }
            }
        }

        private int buildValue(LogicalBuffer buffer, List<AffineExpr> desiredIndices) {
            AffineStatement producer = findProducer(buffer);
            if (producer == null) {
                return addLoad(buffer, desiredIndices);
            }
            ValueKey key = new ValueKey(producer, List.copyOf(desiredIndices));
            Integer cached = cachedValues.get(key);
            if (cached != null) {
                return cached;
            }
            if (!active.add(key)) {
                throw reject("cyclic scalar producer graph at " + producer.name());
            }
            try {
                Map<AffineVariable, AffineExpr> mapping = solveOutputMap(producer, desiredIndices);
                List<AffineAccess> reads = producer.reads();
                List<Integer> inputIds = new ArrayList<>(reads.size());
                for (AffineAccess read : reads) {
                    inputIds.add(buildValue(read.buffer(), substitute(read.indices(), mapping)));
                }
                double factor = producer.kind() == StatementKind.SCALE
                    ? scaleFactor(producer) : Double.NaN;
                ScalarFusionProgram.Opcode opcode = switch (producer.kind()) {
                    case ADD -> ScalarFusionProgram.Opcode.ADD;
                    case SCALE -> ScalarFusionProgram.Opcode.SCALE;
                    case TRANSPOSE -> ScalarFusionProgram.Opcode.TRANSPOSE;
                    default -> throw reject("unsupported scalar producer " + producer.name());
                };
                int id = nodes.size();
                ScalarFusionProgram.ScalarNode node = new ScalarFusionProgram.ScalarNode(
                    id, opcode, inputIds, buffer, List.of(), factor, producer.id());
                nodes.add(node);
                cachedValues.put(key, id);
                return id;
            } finally {
                active.remove(key);
            }
        }

        private ScalarFusionProgram finish(int root) {
            return new ScalarFusionProgram(nodes, root);
        }

        private List<LogicalBuffer> leaves() {
            return List.copyOf(leaves);
        }

        private int addLoad(LogicalBuffer buffer, List<AffineExpr> indices) {
            if (buffer == null) {
                throw reject("scalar load has no buffer");
            }
            validateAccessMap(buffer, indices);
            int id = nodes.size();
            nodes.add(new ScalarFusionProgram.ScalarNode(
                id, ScalarFusionProgram.Opcode.LOAD, List.of(), buffer,
                List.copyOf(indices), Double.NaN, -1));
            if (leafSet.add(buffer)) {
                leaves.add(buffer);
            }
            return id;
        }

        private Map<AffineVariable, AffineExpr> solveOutputMap(
            AffineStatement statement,
            List<AffineExpr> desiredIndices) {
            AffineAccess write = statement.writes().get(0);
            if (write.indices().size() != desiredIndices.size()) {
                throw reject("incompatible access rank at " + statement.name());
            }
            Map<AffineVariable, AffineExpr> result = new HashMap<>();
            for (int dimension = 0; dimension < write.indices().size(); dimension++) {
                AffineExpr outputIndex = write.index(dimension);
                if (!isUnitVariable(outputIndex)) {
                    throw reject("unsupported output affine map at " + statement.name());
                }
                AffineVariable variable = outputIndex.coefficients().firstKey();
                if (!statement.domain().variables().contains(variable)
                    || result.put(variable, desiredIndices.get(dimension)) != null) {
                    throw reject("non-permutation output map at " + statement.name());
                }
                if (!isUnitVariable(desiredIndices.get(dimension))) {
                    throw reject("unsupported composed affine map at " + statement.name());
                }
            }
            if (result.size() != statement.domain().variables().size()
                || !result.keySet().containsAll(statement.domain().variables())) {
                throw reject("output map does not cover statement domain at " + statement.name());
            }
            validateDomainMap(statement.domain(), result);
            return result;
        }

        private void validateAccessMap(LogicalBuffer buffer, List<AffineExpr> indices) {
            if (indices.size() != 2) {
                throw reject("unsupported access rank for %" + buffer.id());
            }
            for (int dimension = 0; dimension < indices.size(); dimension++) {
                AffineExpr expression = indices.get(dimension);
                if (!isUnitVariable(expression)) {
                    throw reject("unsupported affine access map for %" + buffer.id());
                }
                IterationDomain.Range range = range(canonicalDomain,
                    expression.coefficients().firstKey());
                long expectedExtent = dimension == 0
                    ? buffer.shape().rows() : buffer.shape().columns();
                if (range == null || range.lowerInclusive() != 0L
                    || range.upperExclusive() - range.lowerInclusive() != expectedExtent) {
                    throw reject("incompatible domain for access to %" + buffer.id());
                }
            }
        }

        private void validateDomainMap(IterationDomain sourceDomain,
                                       Map<AffineVariable, AffineExpr> mapping) {
            Set<AffineVariable> targets = new HashSet<>();
            for (IterationDomain.Range sourceRange : sourceDomain.ranges()) {
                AffineExpr target = mapping.get(sourceRange.variable());
                if (!isUnitVariable(target)) {
                    throw reject("unsupported composed affine map for " + sourceRange.variable());
                }
                if (!targets.add(target.coefficients().firstKey())) {
                    throw reject("non-permutation composed affine map");
                }
                IterationDomain.Range canonicalRange = range(
                    canonicalDomain, target.coefficients().firstKey());
                if (canonicalRange == null
                    || canonicalRange.lowerInclusive() != sourceRange.lowerInclusive()
                    || canonicalRange.upperExclusive() != sourceRange.upperExclusive()) {
                    throw reject("incompatible iteration domains for composed producer");
                }
            }
        }

        private AffineStatement findProducer(LogicalBuffer buffer) {
            return producers.get(buffer);
        }

        private static List<AffineExpr> substitute(
            List<AffineExpr> expressions,
            Map<AffineVariable, AffineExpr> mapping) {
            List<AffineExpr> result = new ArrayList<>(expressions.size());
            for (AffineExpr expression : expressions) {
                AffineExpr substituted = AffineExpr.constant(expression.constant());
                try {
                    for (Map.Entry<AffineVariable, Long> term : expression.coefficients().entrySet()) {
                        AffineExpr replacement = mapping.get(term.getKey());
                        if (replacement == null) {
                            throw reject("schedule map does not bind " + term.getKey());
                        }
                        substituted = substituted.add(replacement.scale(term.getValue()));
                    }
                } catch (ArithmeticException overflow) {
                    throw reject("affine map composition overflow");
                }
                result.add(substituted);
            }
            return result;
        }

        private static boolean isUnitVariable(AffineExpr expression) {
            return expression != null && expression.constant() == 0L
                && expression.coefficients().size() == 1
                && expression.coefficients().firstEntry().getValue() == 1L;
        }

        private static IterationDomain.Range range(IterationDomain domain,
                                                   AffineVariable variable) {
            return domain.ranges().stream()
                .filter(candidate -> candidate.variable().equals(variable))
                .findFirst().orElse(null);
        }

        private static double scaleFactor(AffineStatement statement) {
            String computation = statement.computation();
            int equals = computation.indexOf('=');
            int multiply = computation.indexOf('*', equals + 1);
            if (equals >= 0 && multiply > equals) {
                try {
                    return Double.parseDouble(computation.substring(equals + 1, multiply).trim());
                } catch (NumberFormatException ignored) {
                    // The rejection below carries the useful planner reason.
                }
            }
            throw reject("cannot recover SCALE factor for " + statement.name());
        }
    }

    private record ValueKey(AffineStatement statement, List<AffineExpr> indices) {
        private ValueKey {
            Objects.requireNonNull(statement, "Scalar statement must not be null");
            Objects.requireNonNull(indices, "Scalar indices must not be null");
        }
    }

    private static FusionRejected reject(String message) {
        return new FusionRejected(message);
    }

    private static final class FusionRejected extends RuntimeException {
        private FusionRejected(String message) {
            super(message);
        }
    }
}
