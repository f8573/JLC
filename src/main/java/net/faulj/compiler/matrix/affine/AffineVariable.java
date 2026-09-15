package net.faulj.compiler.matrix.affine;

import java.util.Objects;

/**
 * A named induction variable in the bounded affine IR.
 *
 * <p>Variables are intentionally just names. M2 does not attach symbolic
 * algebra, runtime values, or schedule decisions to them.</p>
 */
public final class AffineVariable implements Comparable<AffineVariable> {
    private final String name;

    public AffineVariable(String name) {
        if (name == null || !name.matches("[A-Za-z_][A-Za-z0-9_]*")) {
            throw new IllegalArgumentException("Invalid affine variable name: " + name);
        }
        this.name = name;
    }

    public static AffineVariable named(String name) {
        return new AffineVariable(name);
    }

    public String name() {
        return name;
    }

    @Override
    public int compareTo(AffineVariable other) {
        return name.compareTo(other.name);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof AffineVariable variable && name.equals(variable.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return name;
    }
}
