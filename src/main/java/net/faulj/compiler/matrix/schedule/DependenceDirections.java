package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.Dependence;
import net.faulj.compiler.matrix.affine.DependenceKind;
import net.faulj.compiler.matrix.affine.DependenceStatus;

/**
 * Bounded dependence-direction derivation for M2's canonical accesses.
 *
 * <p>This is deliberately not a general affine solver. It compares the
 * emitted access expressions and recognizes constant offsets such as
 * {@code i -> i+1}. Anything else is unknown.</p>
 */
public final class DependenceDirections {
    private DependenceDirections() {
    }

    public static DependenceDirectionInfo analyze(Dependence dependence) {
        if (dependence == null) {
            throw new IllegalArgumentException("Dependence must not be null");
        }
        Set<AffineVariable> variables = new TreeSet<>();
        variables.addAll(dependence.source().domain().variables());
        variables.addAll(dependence.sink().domain().variables());
        Map<AffineVariable, DependenceDirection> directions = new TreeMap<>();
        if (dependence.status() == DependenceStatus.UNKNOWN) {
            for (AffineVariable variable : variables) {
                directions.put(variable, DependenceDirection.UNKNOWN);
            }
            return new DependenceDirectionInfo(dependence, directions, null);
        }

        if (dependence.kind() == DependenceKind.REDUCTION && dependence.reduction() != null) {
            AffineVariable reductionVariable = dependence.reduction().variable();
            for (AffineVariable variable : variables) {
                directions.put(variable, variable.equals(reductionVariable)
                    ? DependenceDirection.LESS : DependenceDirection.EQUAL);
            }
            return new DependenceDirectionInfo(dependence, directions, reductionVariable);
        }

        List<AccessPair> pairs = relevantAccessPairs(dependence);
        if (pairs.isEmpty()) {
            for (AffineVariable variable : variables) {
                directions.put(variable, DependenceDirection.UNKNOWN);
            }
            return new DependenceDirectionInfo(dependence, directions, null);
        }

        for (AffineVariable variable : variables) {
            DependenceDirection combined = null;
            for (AccessPair pair : pairs) {
                DependenceDirection candidate = directionFor(pair.source(), pair.sink(), variable);
                combined = combine(combined, candidate);
            }
            directions.put(variable, combined == null ? DependenceDirection.UNKNOWN : combined);
        }
        return new DependenceDirectionInfo(dependence, directions, null);
    }

    public static DependenceDirectionInfo forDependence(Dependence dependence) {
        return analyze(dependence);
    }

    public static DependenceDirection directionOf(Dependence dependence,
                                                   AffineVariable variable) {
        return analyze(dependence).direction(variable);
    }

    private static List<AccessPair> relevantAccessPairs(Dependence dependence) {
        List<AccessPair> result = new ArrayList<>();
        for (AffineAccess source : dependence.source().accesses()) {
            for (AffineAccess sink : dependence.sink().accesses()) {
                if (source.buffer() != sink.buffer()) {
                    continue;
                }
                boolean relevant = switch (dependence.kind()) {
                    case RAW -> source.writes() && sink.reads();
                    case WAR -> source.reads() && sink.writes();
                    case WAW -> source.writes() && sink.writes();
                    case REDUCTION -> source.isReduction() && sink.isReduction();
                };
                if (relevant) {
                    result.add(new AccessPair(source, sink));
                }
            }
        }
        return result;
    }

    private static DependenceDirection directionFor(AffineAccess source,
                                                     AffineAccess sink,
                                                     AffineVariable variable) {
        if (source.indices().size() != sink.indices().size()) {
            return DependenceDirection.UNKNOWN;
        }
        int sourceDimension = -1;
        int sinkDimension = -1;
        for (int index = 0; index < source.indices().size(); index++) {
            AffineExpr sourceIndex = source.index(index);
            AffineExpr sinkIndex = sink.index(index);
            if (!isSupportedUnitProjection(sourceIndex)
                || !isSupportedUnitProjection(sinkIndex)) {
                return DependenceDirection.UNKNOWN;
            }
            if (sourceIndex.coefficient(variable) == 1L) {
                if (sourceDimension >= 0) {
                    return DependenceDirection.UNKNOWN;
                }
                sourceDimension = index;
            }
            if (sinkIndex.coefficient(variable) == 1L) {
                if (sinkDimension >= 0) {
                    return DependenceDirection.UNKNOWN;
                }
                sinkDimension = index;
            }
        }
        if (sourceDimension < 0 || sourceDimension != sinkDimension) {
            return DependenceDirection.UNKNOWN;
        }
        AffineExpr sourceIndex = source.index(sourceDimension);
        AffineExpr sinkIndex = sink.index(sinkDimension);
        long sourceMinusSink;
        try {
            // sourceIteration + sourceConstant = sinkIteration + sinkConstant
            sourceMinusSink = Math.subtractExact(sinkIndex.constant(), sourceIndex.constant());
        } catch (ArithmeticException overflow) {
            return DependenceDirection.UNKNOWN;
        }
        if (sourceMinusSink == 0L) {
            return DependenceDirection.EQUAL;
        }
        return sourceMinusSink < 0L
            ? DependenceDirection.LESS : DependenceDirection.GREATER;
    }

    private static boolean isSupportedUnitProjection(AffineExpr expression) {
        return expression.coefficients().size() == 1
            && expression.coefficients().firstEntry().getValue() == 1L;
    }

    private static DependenceDirection combine(DependenceDirection current,
                                               DependenceDirection candidate) {
        if (candidate == DependenceDirection.UNKNOWN || current == DependenceDirection.UNKNOWN) {
            return DependenceDirection.UNKNOWN;
        }
        if (current == null || current == DependenceDirection.EQUAL) {
            return candidate;
        }
        if (candidate == DependenceDirection.EQUAL || current == candidate) {
            return current;
        }
        return DependenceDirection.UNKNOWN;
    }

    private record AccessPair(AffineAccess source, AffineAccess sink) {
    }
}
