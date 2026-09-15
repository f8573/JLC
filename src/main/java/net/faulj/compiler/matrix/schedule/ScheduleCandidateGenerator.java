package net.faulj.compiler.matrix.schedule;

import java.util.ArrayList;
import java.util.List;

/**
 * Finite deterministic structural candidate generator for M3.
 *
 * <p>It emits the initial schedule and at most one-step legal candidates for
 * adjacent interchange, fusion, parallel marking, and vector marking. Tile
 * sizes are explicit transformation inputs, so tiling is intentionally not
 * guessed here. There is no recursive or open-ended search.</p>
 */
public final class ScheduleCandidateGenerator {
    public static final int MAX_CANDIDATES = 32;

    private ScheduleCandidateGenerator() {
    }

    public static ScheduleCandidates generate(
        net.faulj.compiler.matrix.affine.AffineProgram program) {
        if (program == null) {
            throw new IllegalArgumentException("Affine program must not be null");
        }
        return generate(SchedulePlan.initial(program));
    }

    public static ScheduleCandidates generate(SchedulePlan base) {
        if (base == null) {
            throw new IllegalArgumentException("Schedule plan must not be null");
        }
        List<ScheduleCandidate> candidates = new ArrayList<>();
        List<String> dumps = new ArrayList<>();
        add(candidates, dumps, new ScheduleCandidate("initial", base));

        boolean full = false;
        for (ScheduleRegion region : base.regions()) {
            int statementId = region.primaryStatementId();
            List<ScheduleLoop> loops = region.band().loops();
            for (int index = 0; index + 1 < loops.size() && !full; index++) {
                ScheduleLoop first = loops.get(index);
                ScheduleLoop second = loops.get(index + 1);
                ScheduleTransformResult attempt = base.interchange(
                    statementId, first.inductionVariable(), second.inductionVariable());
                if (attempt.accepted()) {
                    add(candidates, dumps, new ScheduleCandidate(
                        "interchange(" + first.inductionVariable() + ","
                            + second.inductionVariable() + ") in S" + statementId,
                        attempt.schedule()));
                }
                full = candidates.size() >= MAX_CANDIDATES;
            }
            for (ScheduleLoop loop : loops) {
                if (full) {
                    break;
                }
                ScheduleTransformResult parallel = base.parallel(
                    statementId, loop.inductionVariable());
                if (parallel.accepted()) {
                    add(candidates, dumps, new ScheduleCandidate(
                        "parallel(" + loop.inductionVariable() + ") in S" + statementId,
                        parallel.schedule()));
                }
                full = candidates.size() >= MAX_CANDIDATES;
                if (full) {
                    break;
                }
                ScheduleTransformResult vector = base.vector(
                    statementId, loop.inductionVariable());
                if (vector.accepted()) {
                    add(candidates, dumps, new ScheduleCandidate(
                        "vector(" + loop.inductionVariable() + ") in S" + statementId,
                        vector.schedule()));
                }
                full = candidates.size() >= MAX_CANDIDATES;
            }
            if (full) {
                break;
            }
        }

        for (int index = 0; index + 1 < base.regions().size() && !full; index++) {
            ScheduleRegion first = base.region(index);
            ScheduleRegion second = base.region(index + 1);
            ScheduleTransformResult fusion = base.fusion(
                first.primaryStatementId(), second.primaryStatementId());
            if (fusion.accepted()) {
                add(candidates, dumps, new ScheduleCandidate(
                    "fusion(S" + first.primaryStatementId() + ",S"
                        + second.primaryStatementId() + ")",
                    fusion.schedule()));
            }
            full = candidates.size() >= MAX_CANDIDATES;
        }
        return new ScheduleCandidates(candidates, MAX_CANDIDATES);
    }

    private static void add(List<ScheduleCandidate> candidates,
                            List<String> dumps,
                            ScheduleCandidate candidate) {
        if (candidates.size() >= MAX_CANDIDATES) {
            return;
        }
        String dump = candidate.schedule().dump();
        if (dumps.contains(dump)) {
            return;
        }
        candidates.add(candidate);
        dumps.add(dump);
    }
}
