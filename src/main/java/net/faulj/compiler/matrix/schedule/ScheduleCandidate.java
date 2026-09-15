package net.faulj.compiler.matrix.schedule;

import java.util.Objects;

/**
 * One finite, inspectable schedule candidate.
 */
public record ScheduleCandidate(String label, SchedulePlan schedule) {
    public ScheduleCandidate {
        if (label == null || label.isBlank()) {
            throw new IllegalArgumentException("Candidate label must not be blank");
        }
        Objects.requireNonNull(schedule, "Candidate schedule must not be null");
    }
}
