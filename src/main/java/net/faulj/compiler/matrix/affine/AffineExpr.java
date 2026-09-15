package net.faulj.compiler.matrix.affine;

import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Immutable integer-linear expression.
 *
 * <p>An expression is a constant plus a sorted map of variable coefficients.
 * There is deliberately no operation for multiplying two expressions, so
 * nonlinear terms such as {@code i*j} and {@code i*i} cannot be represented.
 * This is a bounded M2 representation, not a symbolic algebra system.</p>
 */
public final class AffineExpr {
    private final long constant;
    private final NavigableMap<AffineVariable, Long> coefficients;

    public AffineExpr(long constant, Map<AffineVariable, Long> coefficients) {
        this.constant = constant;
        TreeMap<AffineVariable, Long> normalized = new TreeMap<>();
        if (coefficients != null) {
            for (Map.Entry<AffineVariable, Long> entry : coefficients.entrySet()) {
                AffineVariable variable = entry.getKey();
                Long coefficient = entry.getValue();
                if (variable == null || coefficient == null) {
                    throw new IllegalArgumentException("Affine coefficients must not contain nulls");
                }
                if (coefficient == Long.MIN_VALUE) {
                    throw new IllegalArgumentException("Affine coefficient is out of supported range");
                }
                if (coefficient != 0L) {
                    normalized.merge(variable, coefficient, AffineExpr::addExact);
                }
            }
        }
        normalized.values().removeIf(value -> value == 0L);
        this.coefficients = Collections.unmodifiableNavigableMap(normalized);
    }

    private AffineExpr(long constant, NavigableMap<AffineVariable, Long> coefficients, boolean trusted) {
        this.constant = constant;
        this.coefficients = Collections.unmodifiableNavigableMap(new TreeMap<>(coefficients));
    }

    public static AffineExpr constant(long value) {
        return new AffineExpr(value, Map.of());
    }

    public static AffineExpr variable(AffineVariable variable) {
        if (variable == null) {
            throw new IllegalArgumentException("Affine variable must not be null");
        }
        return new AffineExpr(0L, Map.of(variable, 1L));
    }

    public static AffineExpr of(long constant, Map<AffineVariable, Long> coefficients) {
        return new AffineExpr(constant, coefficients);
    }

    public long constant() {
        return constant;
    }

    public NavigableMap<AffineVariable, Long> coefficients() {
        return coefficients;
    }

    public long coefficient(AffineVariable variable) {
        if (variable == null) {
            return 0L;
        }
        return coefficients.getOrDefault(variable, 0L);
    }

    public boolean isConstant() {
        return coefficients.isEmpty();
    }

    public AffineExpr add(AffineExpr other) {
        requireOther(other);
        TreeMap<AffineVariable, Long> result = new TreeMap<>(coefficients);
        for (Map.Entry<AffineVariable, Long> entry : other.coefficients.entrySet()) {
            result.merge(entry.getKey(), entry.getValue(), AffineExpr::addExact);
        }
        result.values().removeIf(value -> value == 0L);
        return new AffineExpr(addExact(constant, other.constant), result, true);
    }

    public AffineExpr plus(AffineExpr other) {
        return add(other);
    }

    public AffineExpr add(long value) {
        return new AffineExpr(addExact(constant, value), coefficients, true);
    }

    public AffineExpr plus(long value) {
        return add(value);
    }

    public AffineExpr subtract(AffineExpr other) {
        requireOther(other);
        return add(other.scale(-1L));
    }

    public AffineExpr scale(long factor) {
        if (factor == Long.MIN_VALUE) {
            throw new IllegalArgumentException("Affine scale is out of supported range");
        }
        TreeMap<AffineVariable, Long> result = new TreeMap<>();
        for (Map.Entry<AffineVariable, Long> entry : coefficients.entrySet()) {
            long coefficient = Math.multiplyExact(entry.getValue(), factor);
            if (coefficient != 0L) {
                result.put(entry.getKey(), coefficient);
            }
        }
        return new AffineExpr(Math.multiplyExact(constant, factor), result, true);
    }

    public AffineExpr multiply(long factor) {
        return scale(factor);
    }

    private static void requireOther(AffineExpr expression) {
        if (expression == null) {
            throw new IllegalArgumentException("Affine expression must not be null");
        }
    }

    private static long addExact(long left, long right) {
        return Math.addExact(left, right);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof AffineExpr expression
            && constant == expression.constant
            && coefficients.equals(expression.coefficients);
    }

    @Override
    public int hashCode() {
        return Objects.hash(constant, coefficients);
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        boolean first = true;
        for (Map.Entry<AffineVariable, Long> entry : coefficients.entrySet()) {
            long coefficient = entry.getValue();
            if (first) {
                appendFirstTerm(result, coefficient, entry.getKey());
                first = false;
            } else {
                appendFollowingTerm(result, coefficient, entry.getKey());
            }
        }
        if (constant != 0L || first) {
            if (first) {
                result.append(constant);
            } else if (constant >= 0L) {
                result.append(" + ").append(constant);
            } else {
                result.append(" - ").append(positiveMagnitude(constant));
            }
        }
        return result.toString();
    }

    private static void appendFirstTerm(StringBuilder result,
                                        long coefficient,
                                        AffineVariable variable) {
        if (coefficient < 0L) {
            result.append('-');
        }
        appendMagnitudeAndVariable(result, coefficient, variable);
    }

    private static void appendFollowingTerm(StringBuilder result,
                                            long coefficient,
                                            AffineVariable variable) {
        if (coefficient < 0L) {
            result.append(" - ");
        } else {
            result.append(" + ");
        }
        appendMagnitudeAndVariable(result, coefficient, variable);
    }

    private static void appendMagnitudeAndVariable(StringBuilder result,
                                                   long coefficient,
                                                   AffineVariable variable) {
        long magnitude = positiveMagnitude(coefficient);
        if (magnitude != 1L) {
            result.append(magnitude).append('*');
        }
        result.append(variable);
    }

    private static long positiveMagnitude(long value) {
        if (value == Long.MIN_VALUE) {
            throw new IllegalArgumentException("Affine value is out of supported range");
        }
        return value < 0L ? -value : value;
    }
}
