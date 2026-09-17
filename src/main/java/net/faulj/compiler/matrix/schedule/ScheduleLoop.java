package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;

/**
 * Immutable loop specification in a schedule band.
 *
 * <p>{@code inductionVariable} is the schedule iterator. A tiled inner loop
 * can additionally bind an original affine variable through
 * {@code valueExpression}; this keeps strip-mining explicit without turning
 * the schedule into executable code.</p>
 */
public final class ScheduleLoop {
    private final AffineVariable inductionVariable;
    private final AffineVariable semanticVariable;
    private final long lowerBound;
    private final long upperBound;
    private final long step;
    private final List<ScheduleAnnotation> annotations;
    private final AffineExpr valueExpression;
    private final String guard;
    private final String description;

    public ScheduleLoop(AffineVariable variable,
                        long lowerBound,
                        long upperBound,
                        long step) {
        this(variable, variable, lowerBound, upperBound, step,
            List.of(), AffineExpr.variable(variable), null, null);
    }

    public ScheduleLoop(AffineVariable variable,
                        long lowerBound,
                        long upperBound) {
        this(variable, lowerBound, upperBound, 1L);
    }

    public ScheduleLoop(AffineVariable inductionVariable,
                        AffineVariable semanticVariable,
                        long lowerBound,
                        long upperBound,
                        long step,
                        List<ScheduleAnnotation> annotations,
                        AffineExpr valueExpression,
                        String guard,
                        String description) {
        this.inductionVariable = Objects.requireNonNull(
            inductionVariable, "Schedule induction variable must not be null");
        this.semanticVariable = Objects.requireNonNull(
            semanticVariable, "Schedule semantic variable must not be null");
        if (upperBound < lowerBound) {
            throw new IllegalArgumentException("Schedule upper bound must not be below lower bound");
        }
        if (step <= 0L) {
            throw new IllegalArgumentException("Schedule loop step must be positive");
        }
        if (annotations == null) {
            throw new IllegalArgumentException("Schedule annotations must not be null");
        }
        List<ScheduleAnnotation> copy = new ArrayList<>(annotations);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Schedule annotations must not contain nulls");
        }
        copy.sort(Comparator.comparingInt(ScheduleAnnotation::ordinal));
        for (int index = 0; index < copy.size(); index++) {
            if (copy.indexOf(copy.get(index)) != index) {
                throw new IllegalArgumentException("Schedule annotations must be unique");
            }
        }
        if (valueExpression == null && guard != null) {
            throw new IllegalArgumentException("A schedule guard requires a value binding");
        }
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.step = step;
        this.annotations = Collections.unmodifiableList(copy);
        this.valueExpression = valueExpression;
        this.guard = guard;
        this.description = description;
    }

    public static ScheduleLoop of(AffineVariable variable,
                                  long lowerBound,
                                  long upperBound) {
        return new ScheduleLoop(variable, lowerBound, upperBound, 1L);
    }

    public AffineVariable variable() {
        return inductionVariable;
    }

    public AffineVariable inductionVariable() {
        return inductionVariable;
    }

    /**
     * Original affine variable represented by this schedule dimension.
     */
    public AffineVariable semanticVariable() {
        return semanticVariable;
    }

    public long lowerBound() {
        return lowerBound;
    }

    public long lower() {
        return lowerBound;
    }

    public long upperBound() {
        return upperBound;
    }

    public long upper() {
        return upperBound;
    }

    public long step() {
        return step;
    }

    public List<ScheduleAnnotation> annotations() {
        return annotations;
    }

    public boolean hasAnnotation(ScheduleAnnotation annotation) {
        return annotations.contains(annotation);
    }

    /**
     * Binding for the original affine variable, or {@code null} for an
     * outer tile iterator that does not itself select an element.
     */
    public AffineExpr valueExpression() {
        return valueExpression;
    }

    public AffineExpr binding() {
        return valueExpression;
    }

    public String guard() {
        return guard;
    }

    public boolean isBoundedByGuard() {
        return guard != null;
    }

    public String description() {
        return description;
    }

    public ScheduleLoop withAnnotation(ScheduleAnnotation annotation) {
        Objects.requireNonNull(annotation, "Schedule annotation must not be null");
        if (annotations.contains(annotation)) {
            return this;
        }
        List<ScheduleAnnotation> updated = new ArrayList<>(annotations);
        updated.add(annotation);
        return copyWith(updated);
    }

    public ScheduleLoop withAnnotations(List<ScheduleAnnotation> updated) {
        return new ScheduleLoop(
            inductionVariable,
            semanticVariable,
            lowerBound,
            upperBound,
            step,
            updated,
            valueExpression,
            guard,
            description);
    }

    public ScheduleLoop copyAs(AffineVariable newInductionVariable,
                               AffineVariable newSemanticVariable,
                               long newLowerBound,
                               long newUpperBound,
                               long newStep,
                               List<ScheduleAnnotation> newAnnotations,
                               AffineExpr newValueExpression,
                               String newGuard,
                               String newDescription) {
        return new ScheduleLoop(
            newInductionVariable,
            newSemanticVariable,
            newLowerBound,
            newUpperBound,
            newStep,
            newAnnotations,
            newValueExpression,
            newGuard,
            newDescription);
    }

    public String dumpHeader() {
        StringBuilder result = new StringBuilder();
        if (!annotations.isEmpty()) {
            result.append('[').append(annotations.stream()
                .map(ScheduleAnnotation::displayName)
                .collect(Collectors.joining(", "))).append("] ");
        }
        result.append("for ").append(inductionVariable)
            .append(" [").append(lowerBound).append(',').append(upperBound).append(")")
            .append(" step ").append(step);
        if (description != null && !description.isBlank()) {
            result.append(" (").append(description).append(')');
        }
        if (valueExpression != null
            && !valueExpression.equals(AffineExpr.variable(inductionVariable))) {
            result.append(" binds ").append(semanticVariable)
                .append(" = ").append(valueExpression);
        }
        if (guard != null && !guard.isBlank()) {
            result.append(" guard(").append(guard).append(')');
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dumpHeader();
    }

    private ScheduleLoop copyWith(List<ScheduleAnnotation> updated) {
        return new ScheduleLoop(
            inductionVariable,
            semanticVariable,
            lowerBound,
            upperBound,
            step,
            updated,
            valueExpression,
            guard,
            description);
    }
}
