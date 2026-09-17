package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AliasRelation;
import net.faulj.compiler.matrix.affine.AliasAnalysis;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;

/** Immutable single-entry portable kernel function. */
public final class KernelFunction {
    private final int id;
    private final String name;
    private final List<KernelBuffer> inputBuffers;
    private final List<KernelBuffer> outputBuffers;
    private final List<KernelBuffer> buffers;
    private final IterationDomain iterationDomain;
    private final List<KernelLoop> loops;
    private final KernelBlock body;
    private final List<KernelValue> values;
    private final KernelProvenance provenance;
    private final List<KernelAliasFact> aliasFacts;

    public KernelFunction(int id,
                          String name,
                          List<KernelBuffer> inputBuffers,
                          List<KernelBuffer> outputBuffers,
                          IterationDomain iterationDomain,
                          List<KernelLoop> loops,
                          KernelBlock body,
                          List<KernelValue> values,
                          KernelProvenance provenance) {
        this(id, name, inputBuffers, outputBuffers, iterationDomain, loops, body, values,
            provenance, conservativeAliasFacts(inputBuffers, outputBuffers));
    }

    public KernelFunction(int id,
                          String name,
                          List<KernelBuffer> inputBuffers,
                          List<KernelBuffer> outputBuffers,
                          IterationDomain iterationDomain,
                          List<KernelLoop> loops,
                          KernelBlock body,
                          List<KernelValue> values,
                          KernelProvenance provenance,
                          List<KernelAliasFact> aliasFacts) {
        this.id = id;
        this.name = name;
        this.inputBuffers = immutable(inputBuffers, "Kernel input buffers");
        this.outputBuffers = immutable(outputBuffers, "Kernel output buffers");
        List<KernelBuffer> combined = new ArrayList<>(this.inputBuffers.size()
            + this.outputBuffers.size());
        combined.addAll(this.inputBuffers);
        combined.addAll(this.outputBuffers);
        this.buffers = Collections.unmodifiableList(combined);
        this.iterationDomain = iterationDomain;
        this.loops = immutable(loops, "Kernel loops");
        this.body = body;
        this.values = immutable(values, "Kernel values");
        this.provenance = provenance;
        this.aliasFacts = immutable(aliasFacts, "Kernel alias facts");
    }

    public int id() {
        return id;
    }

    public String name() {
        return name;
    }

    public List<KernelBuffer> inputBuffers() {
        return inputBuffers;
    }

    public List<KernelBuffer> outputs() {
        return outputBuffers;
    }

    public List<KernelBuffer> outputBuffers() {
        return outputBuffers;
    }

    public List<KernelBuffer> buffers() {
        return buffers;
    }

    public IterationDomain iterationDomain() {
        return iterationDomain;
    }

    public List<KernelLoop> loops() {
        return loops;
    }

    public KernelBlock body() {
        return body;
    }

    public List<KernelValue> values() {
        return values;
    }

    public KernelProvenance provenance() {
        return provenance;
    }

    public List<KernelAliasFact> aliasFacts() {
        return aliasFacts;
    }

    /** Return the explicit fact, or MAY_ALIAS when no fact was supplied. */
    public AliasRelation aliasRelation(KernelBuffer first, KernelBuffer second) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("Kernel alias buffers must not be null");
        }
        for (KernelAliasFact fact : aliasFacts) {
            if ((fact.first() == first && fact.second() == second)
                || (fact.first() == second && fact.second() == first)) {
                return fact.relation();
            }
        }
        return AliasRelation.MAY_ALIAS;
    }

    /** Deterministic textual rendering of this function. */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("function ").append(name).append(" fp64\n");
        result.append("source:\n");
        if (provenance == null) {
            result.append("  (none)\n");
        } else {
            result.append("  fused-region=F").append(provenance.fusedRegionId()).append('\n');
            result.append("  statements=").append(formatStatements(
                provenance.sourceStatementIds())).append('\n');
            result.append("  scalar-nodes=").append(provenance.sourceScalarNodeIds()).append('\n');
            result.append("  eliminated=").append(formatBuffers(
                provenance.eliminatedBufferIds())).append('\n');
            if (provenance.legalityEvidence() != null) {
                result.append("  legality=").append(provenance.legalityEvidence()).append('\n');
            }
            if (provenance.profitabilityEvidence() != null) {
                result.append("  profitability=").append(
                    provenance.profitabilityEvidence()).append('\n');
            }
        }
        result.append("buffers:\n");
        if (buffers.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (KernelBuffer buffer : buffers) {
                result.append("  %").append(buffer.id()).append(' ').append(buffer.name())
                    .append(' ').append(buffer.role().name().toLowerCase())
                    .append(" class=").append(buffer.storageClass())
                    .append(" storage=").append(buffer.storageType())
                    .append(" compute=").append(buffer.computeType())
                    .append(" accumulator=").append(buffer.accumulatorType())
                    .append(" memory=").append(buffer.memorySpace().name().toLowerCase())
                    .append(" layout=").append(buffer.layout().name().toLowerCase())
                    .append(" ownership=").append(buffer.ownership().name().toLowerCase())
                    .append('\n');
            }
        }
        result.append("aliases:\n");
        if (aliasFacts.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (KernelAliasFact fact : aliasFacts) {
                result.append("  ").append(fact).append('\n');
            }
        }
        result.append("loops:\n");
        if (loops.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (KernelLoop loop : loops) {
                result.append("  for ").append(loop.inductionVariable()).append(" = ")
                    .append(loop.lowerBound()).append("..").append(loop.upperBound())
                    .append(" step ").append(loop.step()).append('\n');
            }
        }
        result.append("body:\n");
        if (body == null || body.operations().isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (KernelOp operation : body.operations()) {
                result.append("  ").append(formatOperation(operation)).append('\n');
            }
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private static String formatOperation(KernelOp operation) {
        if (operation == null || operation.opcode() == null) {
            return "<invalid operation>";
        }
        return switch (operation.opcode()) {
            case LOAD -> "k" + operation.resultValueId() + " = load "
                + operation.access();
            case CONSTANT -> "k" + operation.resultValueId() + " = const "
                + Double.toString(operation.immediate());
            case ADD -> "k" + operation.resultValueId() + " = add k"
                + operation.operands().get(0) + ", k" + operation.operands().get(1);
            case MUL -> "k" + operation.resultValueId() + " = mul k"
                + operation.operands().get(0) + ", k" + operation.operands().get(1);
            case STORE -> "store " + operation.access() + ", k"
                + operation.operands().get(0);
        };
    }

    private static String formatStatements(List<Integer> ids) {
        if (ids == null) {
            return "null";
        }
        List<String> result = new ArrayList<>(ids.size());
        for (Integer id : ids) {
            result.add("S" + id);
        }
        return result.toString();
    }

    private static String formatBuffers(List<Integer> ids) {
        if (ids == null) {
            return "null";
        }
        List<String> result = new ArrayList<>(ids.size());
        for (Integer id : ids) {
            result.add("%" + id);
        }
        return result.toString();
    }

    private static List<KernelAliasFact> conservativeAliasFacts(
        List<KernelBuffer> inputs,
        List<KernelBuffer> outputs) {
        if (inputs == null || outputs == null) {
            throw new IllegalArgumentException("Kernel buffers must not be null");
        }
        List<KernelBuffer> buffers = new ArrayList<>(inputs.size() + outputs.size());
        buffers.addAll(inputs);
        buffers.addAll(outputs);
        List<KernelAliasFact> result = new ArrayList<>();
        for (int first = 0; first < buffers.size(); first++) {
            for (int second = first + 1; second < buffers.size(); second++) {
                KernelBuffer left = buffers.get(first);
                KernelBuffer right = buffers.get(second);
                LogicalBuffer leftLogical = left == null ? null : left.logicalBuffer();
                LogicalBuffer rightLogical = right == null ? null : right.logicalBuffer();
                AliasRelation relation = leftLogical == null || rightLogical == null
                    ? AliasRelation.MAY_ALIAS
                    : AliasAnalysis.between(leftLogical, rightLogical);
                result.add(new KernelAliasFact(left, right, relation));
            }
        }
        return result;
    }

    private static <T> List<T> immutable(List<T> source, String role) {
        if (source == null || source.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(role + " must not be null or contain nulls");
        }
        return Collections.unmodifiableList(new ArrayList<>(source));
    }
}
