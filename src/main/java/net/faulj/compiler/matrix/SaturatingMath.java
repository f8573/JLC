package net.faulj.compiler.matrix;

/**
 * Small helpers for non-negative planner arithmetic.
 *
 * <p>Matrix dimensions are ints, but a matrix-chain cost can be much larger
 * than an int and can exceed the range of a long for deliberately extreme
 * symbolic shapes. Planner arithmetic therefore saturates at
 * {@link Long#MAX_VALUE} instead of wrapping.</p>
 */
final class SaturatingMath {
    private SaturatingMath() {
    }

    static long add(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("Saturating planner arithmetic requires non-negative values");
        }
        if (Long.MAX_VALUE - left < right) {
            return Long.MAX_VALUE;
        }
        return left + right;
    }

    static long multiply(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("Saturating planner arithmetic requires non-negative values");
        }
        if (left == 0 || right == 0) {
            return 0;
        }
        if (left > Long.MAX_VALUE / right) {
            return Long.MAX_VALUE;
        }
        return left * right;
    }
}
