package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Bounded result of structural candidate generation.
 */
public final class ScheduleCandidates {
    private final List<ScheduleCandidate> candidates;
    private final int maximumAllowed;

    ScheduleCandidates(List<ScheduleCandidate> candidates, int maximumAllowed) {
        if (candidates == null || candidates.stream().anyMatch(candidate -> candidate == null)) {
            throw new IllegalArgumentException("Candidates must be non-null");
        }
        if (maximumAllowed <= 0 || candidates.size() > maximumAllowed) {
            throw new IllegalArgumentException("Candidate count exceeds configured bound");
        }
        this.candidates = Collections.unmodifiableList(new ArrayList<>(candidates));
        this.maximumAllowed = maximumAllowed;
    }

    public List<ScheduleCandidate> candidates() {
        return candidates;
    }

    public List<ScheduleCandidate> plansWithLabels() {
        return candidates;
    }

    public List<SchedulePlan> schedules() {
        return candidates.stream().map(ScheduleCandidate::schedule).toList();
    }

    public int count() {
        return candidates.size();
    }

    public int size() {
        return count();
    }

    public int maximumAllowed() {
        return maximumAllowed;
    }

    public boolean isWithinBound() {
        return count() <= maximumAllowed;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("candidates (" + count() + "/"
            + maximumAllowed + "):\n");
        for (ScheduleCandidate candidate : candidates) {
            result.append("  ").append(candidate.label()).append('\n');
        }
        return result.toString();
    }
}
