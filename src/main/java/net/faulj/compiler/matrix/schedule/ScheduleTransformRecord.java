package net.faulj.compiler.matrix.schedule;

import java.util.Objects;

/**
 * One deterministic entry in a transformed schedule's history.
 */
public record ScheduleTransformRecord(ScheduleTransformKind kind,
                                      String operation,
                                      LegalityStatus status,
                                      String explanation) {
    public ScheduleTransformRecord {
        Objects.requireNonNull(kind, "Transformation kind must not be null");
        if (operation == null || operation.isBlank()) {
            throw new IllegalArgumentException("Transformation operation must not be blank");
        }
        Objects.requireNonNull(status, "Transformation status must not be null");
        if (explanation == null || explanation.isBlank()) {
            throw new IllegalArgumentException("Transformation explanation must not be blank");
        }
    }

    @Override
    public String toString() {
        return operation + " -> " + status + ": " + explanation;
    }
}
