package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineStatement;

/**
 * A schedule region whose body contains one or more statements under one
 * compatible loop band. Fusion combines adjacent regions into one region.
 */
public final class ScheduleRegion {
    private final List<AffineStatement> statements;
    private final ScheduleBand band;

    public ScheduleRegion(List<AffineStatement> statements, ScheduleBand band) {
        if (statements == null) {
            throw new IllegalArgumentException("Schedule region statements must not be null");
        }
        List<AffineStatement> copy = new ArrayList<>(statements);
        if (copy.isEmpty()) {
            throw new IllegalArgumentException("Schedule region must contain a statement");
        }
        IdentityHashMap<AffineStatement, Boolean> seen = new IdentityHashMap<>();
        for (AffineStatement statement : copy) {
            if (statement == null || seen.put(statement, Boolean.TRUE) != null) {
                throw new IllegalArgumentException(
                    "Schedule region statements must be non-null and unique");
            }
        }
        this.statements = Collections.unmodifiableList(copy);
        this.band = Objects.requireNonNull(band, "Schedule region band must not be null");
        validateBodyStatements();
    }

    public List<AffineStatement> statements() {
        return statements;
    }

    public List<AffineStatement> scheduledStatements() {
        return statements;
    }

    public ScheduleBand band() {
        return band;
    }

    public ScheduleBand schedule() {
        return band;
    }

    public int primaryStatementId() {
        return statements.get(0).id();
    }

    public boolean containsStatement(int statementId) {
        return statements.stream().anyMatch(statement -> statement.id() == statementId);
    }

    public boolean contains(AffineStatement statement) {
        return statements.stream().anyMatch(candidate -> candidate == statement);
    }

    public boolean isSingleton() {
        return statements.size() == 1;
    }

    public ScheduleRegion withBand(ScheduleBand updatedBand) {
        return new ScheduleRegion(statements, updatedBand);
    }

    @Override
    public String toString() {
        return "region " + statementNames() + ":\n" + indent(band.dump());
    }

    private void validateBodyStatements() {
        List<AffineStatement> bodyStatements = new ArrayList<>();
        collectStatements(band.body(), bodyStatements);
        if (bodyStatements.size() != statements.size()) {
            throw new IllegalArgumentException("Schedule region body must contain each region statement once");
        }
        for (int index = 0; index < statements.size(); index++) {
            if (bodyStatements.get(index) != statements.get(index)) {
                throw new IllegalArgumentException(
                    "Schedule region body statement order must match region statements");
            }
        }
    }

    private static void collectStatements(ScheduleNode node,
                                          List<AffineStatement> result) {
        if (node instanceof ScheduleStatement statement) {
            result.add(statement.statement());
        } else if (node instanceof ScheduleSequence sequence) {
            for (ScheduleNode child : sequence.children()) {
                collectStatements(child, result);
            }
        } else {
            collectStatements(((ScheduleBand) node).body(), result);
        }
    }

    private String statementNames() {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < statements.size(); index++) {
            if (index > 0) {
                result.append(',');
            }
            result.append(statements.get(index).name());
        }
        return result.toString();
    }

    private static String indent(String value) {
        return value.replace("\n", "\n  ").stripTrailing();
    }
}
