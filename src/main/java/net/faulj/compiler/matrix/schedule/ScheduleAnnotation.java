package net.faulj.compiler.matrix.schedule;

import java.util.Locale;

/**
 * Explicit metadata attached to a schedule loop.
 */
public enum ScheduleAnnotation {
    PARALLEL,
    VECTOR,
    REDUCTION_PARALLEL_ELIGIBLE,
    REDUCTION_REASSOCIATION;

    public String displayName() {
        return name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
