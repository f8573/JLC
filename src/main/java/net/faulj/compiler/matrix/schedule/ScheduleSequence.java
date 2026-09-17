package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Immutable sequential composition of schedule nodes.
 */
public final class ScheduleSequence implements ScheduleNode {
    private final List<ScheduleNode> children;

    public ScheduleSequence(List<? extends ScheduleNode> children) {
        if (children == null) {
            throw new IllegalArgumentException("Schedule sequence children must not be null");
        }
        List<ScheduleNode> copy = new ArrayList<>(children);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Schedule sequence children must not contain nulls");
        }
        this.children = Collections.unmodifiableList(copy);
    }

    public static ScheduleSequence of(ScheduleNode... children) {
        if (children == null) {
            throw new IllegalArgumentException("Schedule sequence children must not be null");
        }
        return new ScheduleSequence(List.of(children));
    }

    public List<ScheduleNode> children() {
        return children;
    }

    @Override
    public String dump(String indentation) {
        StringBuilder result = new StringBuilder();
        for (ScheduleNode child : children) {
            result.append(child.dump(indentation));
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }
}
