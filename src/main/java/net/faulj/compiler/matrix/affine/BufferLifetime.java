package net.faulj.compiler.matrix.affine;

/**
 * Logical liveness information for one buffer.
 *
 * <p>Statement numbers are deterministic affine-program statement IDs. A
 * {@code null} use means that no consumer use was observed. This record does
 * not authorize early freeing or storage reuse.</p>
 */
public record BufferLifetime(Integer producerStatementId,
                             Integer firstUseStatementId,
                             Integer lastUseStatementId,
                             boolean liveForEvaluation) {
    public BufferLifetime {
        validateId(producerStatementId, "producer statement");
        validateId(firstUseStatementId, "first-use statement");
        validateId(lastUseStatementId, "last-use statement");
        if (firstUseStatementId != null && lastUseStatementId != null
            && firstUseStatementId > lastUseStatementId) {
            throw new IllegalArgumentException("First use must not follow last use");
        }
    }

    public static BufferLifetime none() {
        return new BufferLifetime(null, null, null, false);
    }

    public boolean hasRuntimeLifetime() {
        return liveForEvaluation || producerStatementId != null
            || firstUseStatementId != null || lastUseStatementId != null;
    }

    public Integer producer() {
        return producerStatementId;
    }

    public Integer firstUse() {
        return firstUseStatementId;
    }

    public Integer lastUse() {
        return lastUseStatementId;
    }

    private static void validateId(Integer id, String role) {
        if (id != null && id < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
    }

    @Override
    public String toString() {
        if (!hasRuntimeLifetime()) {
            return "none";
        }
        String producer = producerStatementId == null ? "-" : "S" + producerStatementId;
        String first = firstUseStatementId == null ? "-" : "S" + firstUseStatementId;
        String last = lastUseStatementId == null ? "-" : "S" + lastUseStatementId;
        String span = liveForEvaluation ? ", live=evaluation" : "";
        return "producer=" + producer + ", firstUse=" + first + ", lastUse=" + last + span;
    }
}
