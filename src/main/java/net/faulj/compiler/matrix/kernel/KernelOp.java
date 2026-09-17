package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One immutable portable kernel operation.
 *
 * <p>The general constructor intentionally does only structural copying. The
 * verifier, rather than a constructor side effect, is the authority that
 * diagnoses duplicate IDs, bad operands, and opcode-specific shape errors.</p>
 */
public final class KernelOp {
    private final int resultValueId;
    private final KernelOpcode opcode;
    private final List<Integer> operands;
    private final KernelAccess access;
    private final Double immediate;
    private final int sourceStatementId;
    private final int sourceScalarValueId;

    public KernelOp(int resultValueId,
                    KernelOpcode opcode,
                    List<Integer> operands,
                    KernelAccess access,
                    Double immediate,
                    int sourceStatementId,
                    int sourceScalarValueId) {
        this.resultValueId = resultValueId;
        this.opcode = opcode;
        this.operands = operands == null
            ? null : Collections.unmodifiableList(new ArrayList<>(operands));
        this.access = access;
        this.immediate = immediate;
        this.sourceStatementId = sourceStatementId;
        this.sourceScalarValueId = sourceScalarValueId;
    }

    public static KernelOp load(int resultValueId, KernelAccess access) {
        return load(resultValueId, access, -1, resultValueId);
    }

    public static KernelOp load(int resultValueId,
                                KernelAccess access,
                                int sourceStatementId,
                                int sourceScalarValueId) {
        return new KernelOp(resultValueId, KernelOpcode.LOAD, List.of(), access, null,
            sourceStatementId, sourceScalarValueId);
    }

    public static KernelOp constant(int resultValueId, double value) {
        return constant(resultValueId, value, -1, resultValueId);
    }

    public static KernelOp constant(int resultValueId,
                                    double value,
                                    int sourceStatementId,
                                    int sourceScalarValueId) {
        return new KernelOp(resultValueId, KernelOpcode.CONSTANT, List.of(), null, value,
            sourceStatementId, sourceScalarValueId);
    }

    public static KernelOp add(int resultValueId, int first, int second) {
        return add(resultValueId, first, second, -1, resultValueId);
    }

    public static KernelOp add(int resultValueId,
                               int first,
                               int second,
                               int sourceStatementId,
                               int sourceScalarValueId) {
        return new KernelOp(resultValueId, KernelOpcode.ADD, List.of(first, second), null, null,
            sourceStatementId, sourceScalarValueId);
    }

    public static KernelOp mul(int resultValueId, int first, int second) {
        return mul(resultValueId, first, second, -1, resultValueId);
    }

    public static KernelOp mul(int resultValueId,
                               int first,
                               int second,
                               int sourceStatementId,
                               int sourceScalarValueId) {
        return new KernelOp(resultValueId, KernelOpcode.MUL, List.of(first, second), null, null,
            sourceStatementId, sourceScalarValueId);
    }

    public static KernelOp store(KernelAccess access, int valueId) {
        return store(access, valueId, -1, valueId);
    }

    public static KernelOp store(KernelAccess access,
                                 int valueId,
                                 int sourceStatementId,
                                 int sourceScalarValueId) {
        return new KernelOp(-1, KernelOpcode.STORE, List.of(valueId), access, null,
            sourceStatementId, sourceScalarValueId);
    }

    public int resultValueId() {
        return resultValueId;
    }

    public int resultId() {
        return resultValueId;
    }

    public KernelOpcode opcode() {
        return opcode;
    }

    public List<Integer> operands() {
        return operands;
    }

    public KernelAccess access() {
        return access;
    }

    public Double immediate() {
        return immediate;
    }

    public Double immediateValue() {
        return immediate;
    }

    public int sourceStatementId() {
        return sourceStatementId;
    }

    public int sourceScalarValueId() {
        return sourceScalarValueId;
    }

    public boolean producesValue() {
        return opcode != KernelOpcode.STORE;
    }

    public KernelEffect effect() {
        if (opcode == KernelOpcode.LOAD) {
            return KernelEffect.READS_BUFFER;
        }
        if (opcode == KernelOpcode.STORE) {
            return KernelEffect.WRITES_BUFFER;
        }
        return KernelEffect.PURE;
    }

    @Override
    public String toString() {
        return opcode + " -> " + resultValueId + " " + operands;
    }
}
