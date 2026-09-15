package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Immutable conjunction of half-open rectangular induction-variable ranges.
 *
 * <p>M2 uses static bounds from matrix shapes. It intentionally does not
 * implement Presburger solving, floor division, modulo constraints, or
 * parametric schedule machinery.</p>
 */
public final class IterationDomain {
    /**
     * One half-open rectangular range.
     */
    public record Range(AffineVariable variable, long lowerInclusive, long upperExclusive) {
        public Range {
            if (variable == null) {
                throw new IllegalArgumentException("Domain variable must not be null");
            }
            if (upperExclusive < lowerInclusive) {
                throw new IllegalArgumentException(
                    "Domain upper bound must not be below lower bound for " + variable);
            }
        }

        @Override
        public String toString() {
            return variable + " in [" + lowerInclusive + "," + upperExclusive + ")";
        }
    }

    private final List<Range> ranges;

    public IterationDomain(List<Range> ranges) {
        if (ranges == null) {
            throw new IllegalArgumentException("Domain ranges must not be null");
        }
        List<Range> copy = new ArrayList<>(ranges);
        Set<AffineVariable> seen = new HashSet<>();
        for (Range range : copy) {
            if (range == null || !seen.add(range.variable())) {
                throw new IllegalArgumentException("Domain variables must be non-null and unique");
            }
        }
        this.ranges = Collections.unmodifiableList(copy);
    }

    public static IterationDomain of(Range... ranges) {
        if (ranges == null) {
            throw new IllegalArgumentException("Domain ranges must not be null");
        }
        return new IterationDomain(List.of(ranges));
    }

    public static Range range(AffineVariable variable, long lowerInclusive, long upperExclusive) {
        return new Range(variable, lowerInclusive, upperExclusive);
    }

    public List<Range> ranges() {
        return ranges;
    }

    public List<AffineVariable> variables() {
        return ranges.stream().map(Range::variable).collect(Collectors.toUnmodifiableList());
    }

    public boolean isEmpty() {
        return ranges.isEmpty()
            || ranges.stream().anyMatch(range -> range.lowerInclusive() == range.upperExclusive());
    }

    /** Return the extent of one named range, or {@code -1} when it is absent. */
    public long extent(AffineVariable variable) {
        for (Range range : ranges) {
            if (range.variable().equals(variable)) {
                return range.upperExclusive() - range.lowerInclusive();
            }
        }
        return -1L;
    }

    @Override
    public String toString() {
        if (ranges.isEmpty()) {
            return "{}";
        }
        return ranges.stream().map(Range::toString).collect(Collectors.joining(" && "));
    }
}
