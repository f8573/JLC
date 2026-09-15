package net.faulj.compiler.matrix.schedule;

import java.util.Optional;
import java.util.Objects;

/**
 * Outcome of an immutable schedule transformation attempt.
 *
 * <p>Rejected attempts return the unchanged input schedule and an empty
 * transformed-schedule optional. This keeps UNKNOWN and ILLEGAL outcomes
 * explicit without mutating or partially applying a schedule.</p>
 */
public final class ScheduleTransformResult {
    private final SchedulePlan input;
    private final SchedulePlan schedule;
    private final LegalityResult legality;
    private final ScheduleTransformRecord history;

    ScheduleTransformResult(SchedulePlan input,
                             SchedulePlan schedule,
                             LegalityResult legality,
                             ScheduleTransformRecord history) {
        this.input = Objects.requireNonNull(input, "Input schedule must not be null");
        this.schedule = Objects.requireNonNull(schedule, "Result schedule must not be null");
        this.legality = Objects.requireNonNull(legality, "Legality result must not be null");
        this.history = Objects.requireNonNull(history, "Transformation history must not be null");
    }

    public SchedulePlan input() {
        return input;
    }

    /**
     * The transformed schedule when legal, otherwise the unchanged input
     * schedule.
     */
    public SchedulePlan schedule() {
        return schedule;
    }

    public Optional<SchedulePlan> transformedSchedule() {
        return accepted() ? Optional.of(schedule) : Optional.empty();
    }

    public LegalityResult legality() {
        return legality;
    }

    public LegalityStatus status() {
        return legality.status();
    }

    public String explanation() {
        return legality.explanation();
    }

    public String diagnostic() {
        return explanation();
    }

    public ScheduleTransformRecord history() {
        return history;
    }

    public boolean accepted() {
        return legality.isLegal();
    }

    public boolean rejected() {
        return !accepted();
    }

    @Override
    public String toString() {
        return history.toString();
    }
}
