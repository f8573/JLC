package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable link from one kernel back to its R2 fused region. */
public final class KernelProvenance {
    private final int fusedRegionId;
    private final List<Integer> sourceStatementIds;
    private final List<Integer> sourceScalarNodeIds;
    private final List<Integer> eliminatedBufferIds;
    private final String legalityEvidence;
    private final String profitabilityEvidence;

    public KernelProvenance(int fusedRegionId,
                            List<Integer> sourceStatementIds,
                            List<Integer> sourceScalarNodeIds,
                            List<Integer> eliminatedBufferIds) {
        this(fusedRegionId, sourceStatementIds, sourceScalarNodeIds, eliminatedBufferIds,
            null, null);
    }

    public KernelProvenance(int fusedRegionId,
                            List<Integer> sourceStatementIds,
                            List<Integer> sourceScalarNodeIds,
                            List<Integer> eliminatedBufferIds,
                            String legalityEvidence,
                            String profitabilityEvidence) {
        this.fusedRegionId = fusedRegionId;
        this.sourceStatementIds = immutable(sourceStatementIds);
        this.sourceScalarNodeIds = immutable(sourceScalarNodeIds);
        this.eliminatedBufferIds = immutable(eliminatedBufferIds);
        this.legalityEvidence = legalityEvidence;
        this.profitabilityEvidence = profitabilityEvidence;
    }

    public int fusedRegionId() {
        return fusedRegionId;
    }

    public int sourceFusedRegionId() {
        return fusedRegionId;
    }

    public List<Integer> sourceStatementIds() {
        return sourceStatementIds;
    }

    public List<Integer> sourceScalarNodeIds() {
        return sourceScalarNodeIds;
    }

    public List<Integer> eliminatedBufferIds() {
        return eliminatedBufferIds;
    }

    /** R2 legality proof retained for later diagnostics and backend audits. */
    public String legalityEvidence() {
        return legalityEvidence;
    }

    /** R2 profitability rationale retained without making it an optimization rule. */
    public String profitabilityEvidence() {
        return profitabilityEvidence;
    }

    private static List<Integer> immutable(List<Integer> source) {
        if (source == null) {
            return null;
        }
        return Collections.unmodifiableList(new ArrayList<>(source));
    }
}
