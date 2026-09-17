package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.LogicalBuffer;

/**
 * Semantic scalar SSA-like program used by one R2 fused region.
 *
 * <p>This is deliberately still a high-level representation.  It records
 * loads, scalar arithmetic, and transpose value flow, but has no vector
 * width, register, tile, FMA, or machine opcode concepts.  The executable
 * step compiles this immutable list once and reuses it for every matrix point.</p>
 */
public final class ScalarFusionProgram {
    public enum Opcode {
        LOAD,
        SCALE,
        ADD,
        TRANSPOSE
    }

    /** One deterministic scalar SSA definition. */
    public static final class ScalarNode {
        private final int id;
        private final Opcode opcode;
        private final List<Integer> inputs;
        private final LogicalBuffer buffer;
        private final List<AffineExpr> indices;
        private final double factor;
        private final int statementId;

        public ScalarNode(int id,
                          Opcode opcode,
                          List<Integer> inputs,
                          LogicalBuffer buffer,
                          List<AffineExpr> indices,
                          double factor,
                          int statementId) {
            if (id < 0) {
                throw new IllegalArgumentException("Scalar node ID must be non-negative");
            }
            this.id = id;
            this.opcode = Objects.requireNonNull(opcode, "Scalar opcode must not be null");
            if (inputs == null || inputs.stream().anyMatch(Objects::isNull)
                || inputs.stream().anyMatch(input -> input < 0 || input >= id)) {
                throw new IllegalArgumentException(
                    "Scalar node inputs must reference earlier scalar nodes");
            }
            this.inputs = Collections.unmodifiableList(new ArrayList<>(inputs));
            this.buffer = Objects.requireNonNull(buffer, "Scalar node buffer must not be null");
            if (indices == null || indices.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("Scalar node indices must not be null");
            }
            this.indices = Collections.unmodifiableList(new ArrayList<>(indices));
            this.factor = factor;
            if (statementId < -1) {
                throw new IllegalArgumentException("Scalar statement ID must be -1 or non-negative");
            }
            this.statementId = statementId;
            validateShape();
        }

        public int id() {
            return id;
        }

        public Opcode opcode() {
            return opcode;
        }

        public List<Integer> inputs() {
            return inputs;
        }

        public LogicalBuffer buffer() {
            return buffer;
        }

        public List<AffineExpr> indices() {
            return indices;
        }

        public double factor() {
            return factor;
        }

        public int statementId() {
            return statementId;
        }

        public boolean isLoad() {
            return opcode == Opcode.LOAD;
        }

        private void validateShape() {
            int expectedInputs = switch (opcode) {
                case LOAD -> 0;
                case SCALE, TRANSPOSE -> 1;
                case ADD -> 2;
            };
            if (inputs.size() != expectedInputs) {
                throw new IllegalArgumentException(
                    opcode + " scalar node expected " + expectedInputs
                        + " inputs, got " + inputs.size());
            }
            if (opcode == Opcode.LOAD && indices.size() != 2) {
                throw new IllegalArgumentException("Matrix scalar loads require two indices");
            }
            if (opcode != Opcode.SCALE && !Double.isNaN(factor)) {
                throw new IllegalArgumentException("Only SCALE scalar nodes carry a factor");
            }
            if (opcode == Opcode.SCALE && statementId < 0) {
                throw new IllegalArgumentException("SCALE scalar nodes require a source statement");
            }
        }
    }

    private final List<ScalarNode> nodes;
    private final int rootNodeId;
    private final int[] useCounts;

    public ScalarFusionProgram(List<ScalarNode> nodes, int rootNodeId) {
        if (nodes == null || nodes.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Scalar nodes must not be null");
        }
        List<ScalarNode> copy = new ArrayList<>(nodes);
        for (int index = 0; index < copy.size(); index++) {
            if (copy.get(index).id() != index) {
                throw new IllegalArgumentException("Scalar node IDs must be contiguous");
            }
        }
        if (rootNodeId < 0 || rootNodeId >= copy.size()) {
            throw new IllegalArgumentException("Scalar root node must reference a node");
        }
        this.nodes = Collections.unmodifiableList(copy);
        this.rootNodeId = rootNodeId;
        this.useCounts = new int[copy.size()];
        for (ScalarNode node : copy) {
            for (int input : node.inputs()) {
                useCounts[input]++;
            }
        }
    }

    public List<ScalarNode> nodes() {
        return nodes;
    }

    public int rootNodeId() {
        return rootNodeId;
    }

    public ScalarNode root() {
        return nodes.get(rootNodeId);
    }

    public int nodeCount() {
        return nodes.size();
    }

    /** Number of non-load scalar operations in the region. */
    public int internalNodeCount() {
        return (int) nodes.stream().filter(node -> !node.isLoad()).count();
    }

    /** Number of scalar SSA input edges, including loads used by operations. */
    public int scalarUseCount() {
        int result = 0;
        for (ScalarNode node : nodes) {
            result += node.inputs().size();
        }
        return result;
    }

    /** Number of internal scalar producers consumed by more than one node. */
    public int sharedProducerCount() {
        int result = 0;
        for (ScalarNode node : nodes) {
            if (!node.isLoad() && useCounts[node.id()] > 1) {
                result++;
            }
        }
        return result;
    }

    public int useCount(int nodeId) {
        if (nodeId < 0 || nodeId >= useCounts.length) {
            throw new IllegalArgumentException("Unknown scalar node ID: " + nodeId);
        }
        return useCounts[nodeId];
    }

    /** Deterministic semantic dump suitable for compiler diagnostics. */
    public String dump() {
        StringBuilder result = new StringBuilder("scalar SSA:\n");
        if (nodes.isEmpty()) {
            return result.append("  (none)\n").toString();
        }
        for (ScalarNode node : nodes) {
            result.append("  v").append(node.id()).append(" = ");
            switch (node.opcode()) {
                case LOAD -> result.append("load %").append(node.buffer().id())
                    .append(formatIndices(node.indices()));
                case SCALE -> result.append("scale ").append(Double.toString(node.factor()))
                    .append(" v").append(node.inputs().get(0));
                case ADD -> result.append("add v").append(node.inputs().get(0))
                    .append(" v").append(node.inputs().get(1));
                case TRANSPOSE -> result.append("transpose v").append(node.inputs().get(0));
            }
            if (node.statementId() >= 0) {
                result.append("  [S").append(node.statementId()).append(']');
            }
            if (useCounts[node.id()] > 1) {
                result.append("  [shared x").append(useCounts[node.id()]).append(']');
            }
            result.append('\n');
        }
        result.append("  root=v").append(rootNodeId).append('\n');
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private static String formatIndices(List<AffineExpr> indices) {
        StringBuilder result = new StringBuilder("[");
        for (int index = 0; index < indices.size(); index++) {
            if (index > 0) {
                result.append(',');
            }
            result.append(indices.get(index));
        }
        return result.append(']').toString();
    }
}
