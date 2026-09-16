package net.faulj.compiler.matrix.schedule;

import java.util.Locale;

/**
 * The complete M3 schedule transformation vocabulary.
 */
public enum ScheduleTransformKind {
    INTERCHANGE,
    STRIP_MINE,
    FUSION,
    PARALLEL,
    VECTOR;

    public String displayName() {
        return name().toLowerCase(Locale.ROOT).replace('_', '-');
    }
}
