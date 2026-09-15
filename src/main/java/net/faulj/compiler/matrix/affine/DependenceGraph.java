package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Deterministic, conservative dependence graph for an {@link AffineProgram}.
 *
 * <p>The analyzer is intentionally scoped to canonical accesses emitted by
 * M2. An ambiguous possible alias is represented as {@link
 * DependenceStatus#UNKNOWN}; it is never treated as independence.</p>
 */
public final class DependenceGraph {
    private final List<AffineStatement> statements;
    private final List<Dependence> dependences;
    private final Map<Key, Dependence> byKey;

    public DependenceGraph(List<AffineStatement> statements) {
        this(statements, analyzeDependences(statements));
    }

    private DependenceGraph(List<AffineStatement> statements, List<Dependence> dependences) {
        if (statements == null) {
            throw new IllegalArgumentException("Dependence statements must not be null");
        }
        List<AffineStatement> orderedStatements = new ArrayList<>(statements);
        orderedStatements.sort(Comparator.comparingInt(AffineStatement::id));
        this.statements = Collections.unmodifiableList(orderedStatements);
        this.dependences = Collections.unmodifiableList(new ArrayList<>(dependences));
        Map<Key, Dependence> lookup = new HashMap<>();
        for (Dependence dependence : dependences) {
            lookup.put(new Key(dependence.source(), dependence.sink(), dependence.kind()), dependence);
        }
        this.byKey = Collections.unmodifiableMap(lookup);
    }

    public static DependenceGraph analyze(List<AffineStatement> statements) {
        return new DependenceGraph(statements);
    }

    public List<AffineStatement> statements() {
        return statements;
    }

    /**
     * Returns only proven or unknown relationships. A missing relationship is
     * the explicit {@link DependenceStatus#PROVEN_NONE} result.
     */
    public List<Dependence> dependences() {
        return dependences;
    }

    public List<Dependence> edges() {
        return dependences;
    }

    public List<Dependence> unknownDependences() {
        return dependences.stream()
            .filter(Dependence::isUnknown)
            .collect(Collectors.toUnmodifiableList());
    }

    public List<Dependence> dependencesOf(AffineStatement statement) {
        if (statement == null) {
            throw new IllegalArgumentException("Statement must not be null");
        }
        return dependences.stream()
            .filter(edge -> edge.source() == statement || edge.sink() == statement)
            .collect(Collectors.toUnmodifiableList());
    }

    public DependenceStatus query(AffineStatement source,
                                  AffineStatement sink,
                                  DependenceKind kind) {
        if (source == null || sink == null || kind == null) {
            throw new IllegalArgumentException("Dependence query values must not be null");
        }
        Dependence dependence = byKey.get(new Key(source, sink, kind));
        return dependence == null ? DependenceStatus.PROVEN_NONE : dependence.status();
    }

    public DependenceStatus status(AffineStatement source,
                                   AffineStatement sink,
                                   DependenceKind kind) {
        return query(source, sink, kind);
    }

    public Dependence dependence(AffineStatement source,
                                 AffineStatement sink,
                                 DependenceKind kind) {
        return byKey.get(new Key(source, sink, kind));
    }

    public String dump() {
        StringBuilder result = new StringBuilder("dependences:\n");
        if (dependences.isEmpty()) {
            return result.append("  (none)\n").toString();
        }
        for (Dependence dependence : dependences) {
            result.append("  ").append(dependence).append('\n');
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private static List<Dependence> analyzeDependences(List<AffineStatement> input) {
        if (input == null) {
            throw new IllegalArgumentException("Dependence statements must not be null");
        }
        List<AffineStatement> statements = new ArrayList<>(input);
        statements.sort(Comparator.comparingInt(AffineStatement::id));
        Map<Key, Dependence> result = new LinkedHashMap<>();

        for (int index = 0; index < statements.size(); index++) {
            AffineStatement source = statements.get(index);
            if (source.domain().isEmpty()) {
                continue;
            }
            if (source.kind() == StatementKind.MATMUL_UPDATE && source.hasReduction()
                && source.domain().extent(source.reduction().variable()) >= 2L) {
                add(result, new Dependence(
                    source,
                    source,
                    DependenceKind.REDUCTION,
                    DependenceStatus.PROVEN_DEPENDENCE,
                    "same (i,j); k -> k+1",
                    source.reduction()));
            }
            compareAccesses(source, source, result);
            for (int next = index + 1; next < statements.size(); next++) {
                AffineStatement sink = statements.get(next);
                if (sink.domain().isEmpty()) {
                    continue;
                }
                compareAccesses(source, sink, result);
            }
        }
        return new ArrayList<>(result.values());
    }

    private static void compareAccesses(AffineStatement source,
                                        AffineStatement sink,
                                        Map<Key, Dependence> result) {
        for (AffineAccess sourceAccess : source.accesses()) {
            for (AffineAccess sinkAccess : sink.accesses()) {
                if (source == sink && sourceAccess == sinkAccess) {
                    continue;
                }
                if (sourceAccess.isReduction() && sinkAccess.isReduction()
                    && sourceAccess.buffer() == sinkAccess.buffer()) {
                    continue;
                }

                DependenceStatus overlap = overlapStatus(
                    source, sink, sourceAccess, sinkAccess);
                if (overlap == DependenceStatus.PROVEN_NONE) {
                    continue;
                }
                String relation = overlap == DependenceStatus.UNKNOWN
                    ? "possible overlap requires conservative alias proof"
                    : relationFor(source, sink, sourceAccess, sinkAccess);

                if (sourceAccess.writes() && sinkAccess.reads()) {
                    add(result, new Dependence(
                        source,
                        sink,
                        DependenceKind.RAW,
                        overlap,
                        relation));
                }
                if (sourceAccess.reads() && sinkAccess.writes()) {
                    add(result, new Dependence(
                        source,
                        sink,
                        DependenceKind.WAR,
                        overlap,
                        relation));
                }
                if (sourceAccess.writes() && sinkAccess.writes()) {
                    add(result, new Dependence(
                        source,
                        sink,
                        DependenceKind.WAW,
                        overlap,
                        relation));
                }
            }
        }
    }

    private static DependenceStatus overlapStatus(AffineStatement sourceStatement,
                                                  AffineStatement sinkStatement,
                                                  AffineAccess source,
                                                  AffineAccess sink) {
        AliasRelation alias = AliasAnalysis.between(source.buffer(), sink.buffer());
        if (alias == AliasRelation.NO_ALIAS) {
            return DependenceStatus.PROVEN_NONE;
        }
        if (alias == AliasRelation.MAY_ALIAS) {
            return !source.writes() && !sink.writes()
                ? DependenceStatus.PROVEN_NONE
                : DependenceStatus.UNKNOWN;
        }
        if (isCanonicalMatMulFlow(sourceStatement, sinkStatement, source, sink)) {
            return DependenceStatus.PROVEN_DEPENDENCE;
        }
        return supportedOverlap(sourceStatement, sinkStatement, source, sink);
    }

    private static boolean isCanonicalMatMulFlow(AffineStatement sourceStatement,
                                                  AffineStatement sinkStatement,
                                                  AffineAccess source,
                                                  AffineAccess sink) {
        if (source.buffer() != sink.buffer() || !source.hasSameIndices(sink)) {
            return false;
        }
        return (sourceStatement.kind() == StatementKind.MATMUL_INIT
                && sinkStatement.kind() == StatementKind.MATMUL_UPDATE)
            || (sourceStatement.kind() == StatementKind.MATMUL_UPDATE
                && sourceStatement.hasReduction());
    }

    private static DependenceStatus supportedOverlap(AffineStatement sourceStatement,
                                                      AffineStatement sinkStatement,
                                                      AffineAccess source,
                                                      AffineAccess sink) {
        if (source.indices().size() != sink.indices().size()) {
            return DependenceStatus.UNKNOWN;
        }
        java.util.Set<AffineVariable> sourceVariables = new java.util.HashSet<>();
        java.util.Set<AffineVariable> sinkVariables = new java.util.HashSet<>();
        boolean differentInstances = sourceStatement != sinkStatement;
        for (int dimension = 0; dimension < source.indices().size(); dimension++) {
            AffineExpr left = source.index(dimension);
            AffineExpr right = sink.index(dimension);
            if (!isUnitProjection(left) || !isUnitProjection(right)) {
                return DependenceStatus.UNKNOWN;
            }
            AffineVariable leftVariable = left.coefficients().firstKey();
            AffineVariable rightVariable = right.coefficients().firstKey();
            if (!sourceVariables.add(leftVariable) || !sinkVariables.add(rightVariable)) {
                return DependenceStatus.UNKNOWN;
            }
            IterationDomain.Range leftRange = range(sourceStatement.domain(), leftVariable);
            IterationDomain.Range rightRange = range(sinkStatement.domain(), rightVariable);
            if (leftRange == null || rightRange == null) {
                return DependenceStatus.UNKNOWN;
            }
            long leftLower;
            long leftUpper;
            long rightLower;
            long rightUpper;
            try {
                leftLower = Math.addExact(leftRange.lowerInclusive(), left.constant());
                leftUpper = Math.addExact(leftRange.upperExclusive(), left.constant());
                rightLower = Math.addExact(rightRange.lowerInclusive(), right.constant());
                rightUpper = Math.addExact(rightRange.upperExclusive(), right.constant());
            } catch (ArithmeticException overflow) {
                return DependenceStatus.UNKNOWN;
            }
            if (Math.max(leftLower, rightLower) >= Math.min(leftUpper, rightUpper)) {
                return DependenceStatus.PROVEN_NONE;
            }
            if (!leftVariable.equals(rightVariable) || left.constant() != right.constant()) {
                differentInstances = true;
            }
        }
        if (sourceVariables.size() != sourceStatement.domain().variables().size()
            || sinkVariables.size() != sinkStatement.domain().variables().size()) {
            return DependenceStatus.UNKNOWN;
        }
        return differentInstances
            ? DependenceStatus.PROVEN_DEPENDENCE : DependenceStatus.PROVEN_NONE;
    }

    private static boolean isUnitProjection(AffineExpr expression) {
        return expression.coefficients().size() == 1
            && expression.coefficients().firstEntry().getValue() == 1L;
    }

    private static IterationDomain.Range range(IterationDomain domain,
                                                AffineVariable variable) {
        return domain.ranges().stream()
            .filter(candidate -> candidate.variable().equals(variable))
            .findFirst().orElse(null);
    }

    private static String relationFor(AffineStatement source,
                                      AffineStatement sink,
                                      AffineAccess sourceAccess,
                                      AffineAccess sinkAccess) {
        if (source.kind() == StatementKind.MATMUL_INIT
            && sink.kind() == StatementKind.MATMUL_UPDATE) {
            return "same (i,j); update at k=0";
        }
        if (sourceAccess.hasSameIndices(sinkAccess)) {
            return "same affine indices";
        }
        return "producer output reaches consumer";
    }

    private static void add(Map<Key, Dependence> result, Dependence candidate) {
        Key key = new Key(candidate.source(), candidate.sink(), candidate.kind());
        Dependence previous = result.get(key);
        if (previous == null
            || previous.status() == DependenceStatus.PROVEN_DEPENDENCE
                && candidate.status() == DependenceStatus.UNKNOWN) {
            result.put(key, candidate);
        }
    }

    private record Key(AffineStatement source, AffineStatement sink, DependenceKind kind) {
        private Key {
            Objects.requireNonNull(source);
            Objects.requireNonNull(sink);
            Objects.requireNonNull(kind);
        }
    }
}
