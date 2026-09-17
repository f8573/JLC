package net.faulj.compiler.matrix.schedule;

import java.util.Objects;
import java.util.stream.Collectors;

import net.faulj.compiler.matrix.affine.AffineStatement;

/**
 * Leaf that schedules one immutable affine statement.
 */
public final class ScheduleStatement implements ScheduleNode {
    private final AffineStatement statement;

    public ScheduleStatement(AffineStatement statement) {
        this.statement = Objects.requireNonNull(statement, "Scheduled statement must not be null");
    }

    public AffineStatement statement() {
        return statement;
    }

    public int id() {
        return statement.id();
    }

    public String name() {
        return statement.name();
    }

    @Override
    public String dump(String indentation) {
        String variables = statement.domain().variables().stream()
            .map(variable -> variable.name())
            .collect(Collectors.joining(","));
        return indentation + statement.name() + "[" + variables + "]\n";
    }

    @Override
    public String toString() {
        return dump().stripTrailing();
    }
}
