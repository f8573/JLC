package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Immutable loop band (a deterministic ordered loop nest) around a schedule
 * body. The band is the schedule-tree representation of nested loops.
 */
public final class ScheduleBand implements ScheduleNode {
    private final List<ScheduleLoop> loops;
    private final ScheduleNode body;

    public ScheduleBand(List<ScheduleLoop> loops, ScheduleNode body) {
        if (loops == null) {
            throw new IllegalArgumentException("Schedule band loops must not be null");
        }
        List<ScheduleLoop> copy = new ArrayList<>(loops);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Schedule band loops must not contain nulls");
        }
        this.loops = Collections.unmodifiableList(copy);
        this.body = Objects.requireNonNull(body, "Schedule band body must not be null");
    }

    public List<ScheduleLoop> loops() {
        return loops;
    }

    public ScheduleLoop loop(int index) {
        return loops.get(index);
    }

    public ScheduleNode body() {
        return body;
    }

    public ScheduleBand withLoops(List<ScheduleLoop> updatedLoops) {
        return new ScheduleBand(updatedLoops, body);
    }

    public ScheduleBand withBody(ScheduleNode updatedBody) {
        return new ScheduleBand(loops, updatedBody);
    }

    public int indexOfInductionVariable(String name) {
        if (name == null) {
            return -1;
        }
        for (int index = 0; index < loops.size(); index++) {
            if (loops.get(index).inductionVariable().name().equals(name)) {
                return index;
            }
        }
        return -1;
    }

    public int indexOfInductionVariable(net.faulj.compiler.matrix.affine.AffineVariable variable) {
        if (variable == null) {
            return -1;
        }
        for (int index = 0; index < loops.size(); index++) {
            if (loops.get(index).inductionVariable().equals(variable)) {
                return index;
            }
        }
        return -1;
    }

    @Override
    public String dump(String indentation) {
        StringBuilder result = new StringBuilder();
        String nestedIndentation = indentation;
        for (ScheduleLoop loop : loops) {
            result.append(nestedIndentation).append(loop.dumpHeader()).append('\n');
            nestedIndentation += "  ";
        }
        result.append(body.dump(nestedIndentation));
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }
}
