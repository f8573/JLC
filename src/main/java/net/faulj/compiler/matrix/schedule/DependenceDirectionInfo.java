package net.faulj.compiler.matrix.schedule;

import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;

import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.Dependence;

/**
 * Direction vector information for one M2 dependence.
 */
public final class DependenceDirectionInfo {
    private final Dependence dependence;
    private final NavigableMap<AffineVariable, DependenceDirection> directions;
    private final AffineVariable carriedBy;

    DependenceDirectionInfo(Dependence dependence,
                            Map<AffineVariable, DependenceDirection> directions,
                            AffineVariable carriedBy) {
        this.dependence = Objects.requireNonNull(dependence, "Dependence must not be null");
        TreeMap<AffineVariable, DependenceDirection> copy = new TreeMap<>();
        if (directions == null) {
            throw new IllegalArgumentException("Dependence directions must not be null");
        }
        for (Map.Entry<AffineVariable, DependenceDirection> entry : directions.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                throw new IllegalArgumentException("Dependence directions must not contain nulls");
            }
            copy.put(entry.getKey(), entry.getValue());
        }
        this.directions = Collections.unmodifiableNavigableMap(copy);
        this.carriedBy = carriedBy;
    }

    public Dependence dependence() {
        return dependence;
    }

    public AffineStatement source() {
        return dependence.source();
    }

    public AffineStatement sink() {
        return dependence.sink();
    }

    public NavigableMap<AffineVariable, DependenceDirection> directions() {
        return directions;
    }

    public DependenceDirection direction(AffineVariable variable) {
        if (variable == null) {
            return DependenceDirection.UNKNOWN;
        }
        return directions.getOrDefault(variable, DependenceDirection.UNKNOWN);
    }

    public DependenceDirection directionOf(AffineVariable variable) {
        return direction(variable);
    }

    public Optional<AffineVariable> carriedBy() {
        return Optional.ofNullable(carriedBy);
    }

    public boolean isUnknown() {
        return directions.values().stream().anyMatch(direction -> direction == DependenceDirection.UNKNOWN);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        boolean first = true;
        for (Map.Entry<AffineVariable, DependenceDirection> entry : directions.entrySet()) {
            if (!first) {
                result.append(',');
            }
            result.append(entry.getKey()).append(entry.getValue());
            first = false;
        }
        if (carriedBy != null) {
            result.append(" carriedBy=").append(carriedBy);
        }
        return result.toString();
    }
}
