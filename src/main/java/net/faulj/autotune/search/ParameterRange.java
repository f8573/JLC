package net.faulj.autotune.search;

import java.util.Locale;

/**
 * Represents a constrained range for a single GEMM blocking parameter.
 *
 * <p>All legal values are integers in [{@link #min}, {@link #max}],
 * aligned to {@link #step}. The step ensures alignment (e.g., KC
 * multiples of 8, NC multiples of NR).</p>
 *
 * <p>This is a <b>lazy</b> representation. It does not enumerate or
 * materialize candidate values. Phase 4 queries ranges on demand.</p>
 */
public final class ParameterRange {

    public final String name;
    public final int min;
    public final int max;
    public final int step;

    public ParameterRange(String name, int min, int max, int step) {
        if (step <= 0) throw new IllegalArgumentException("step must be > 0");
        if (min > max) throw new IllegalArgumentException(name + ": min(" + min + ") > max(" + max + ")");
        this.name = name;
        this.min = min;
        this.max = max;
        this.step = step;
    }

    /** True if this range contains exactly one value. */
    public boolean isFixed() {
        return min == max;
    }

    /** The single value if fixed, otherwise throws. */
    public int fixedValue() {
        if (!isFixed()) throw new IllegalStateException(name + " is not fixed: " + this);
        return min;
    }

    /** True if the given value falls within [min, max] and is aligned to step. */
    public boolean contains(int value) {
        return value >= min && value <= max && ((value - min) % step == 0);
    }

    @Override
    public String toString() {
        if (isFixed()) {
            return String.format(Locale.ROOT, "%s = {%d}", name, min);
        }
        if (step == 1) {
            return String.format(Locale.ROOT, "%s in [%d, %d]", name, min, max);
        }
        return String.format(Locale.ROOT, "%s in [%d, %d], step %d", name, min, max, step);
    }
}
