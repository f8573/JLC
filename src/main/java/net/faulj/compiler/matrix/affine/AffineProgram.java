package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.OptimizationSemantics;

/**
 * Immutable deterministic semantic program lowered from one M1
 * {@code ExecutionPlan}.
 *
 * <p>This is inspection and legality-analysis IR only. It intentionally has
 * no interpreter, scheduler, allocator, or native code generator.</p>
 */
public final class AffineProgram {
    private final ExecutionPlan originatingPlan;
    private final OptimizationSemantics semantics;
    private final List<LogicalBuffer> buffers;
    private final List<AffineVariable> variables;
    private final List<AffineStatement> statements;
    private final LogicalBuffer resultBuffer;
    private final DependenceGraph dependenceGraph;
    private final Map<Integer, LogicalBuffer> buffersById;

    /**
     * Convenience entry point for callers starting from an M1 plan.
     */
    public static AffineProgram lower(ExecutionPlan plan) {
        return AffineLowerer.lower(plan);
    }

    public static AffineProgram from(ExecutionPlan plan) {
        return lower(plan);
    }

    public AffineProgram(OptimizationSemantics semantics,
                         List<LogicalBuffer> buffers,
                         List<AffineVariable> variables,
                         List<AffineStatement> statements,
                         LogicalBuffer resultBuffer,
                         DependenceGraph dependenceGraph) {
        this(null, semantics, buffers, variables, statements, resultBuffer, dependenceGraph);
    }

    AffineProgram(ExecutionPlan originatingPlan,
                  OptimizationSemantics semantics,
                  List<LogicalBuffer> buffers,
                  List<AffineVariable> variables,
                  List<AffineStatement> statements,
                  LogicalBuffer resultBuffer,
                  DependenceGraph dependenceGraph) {
        this.originatingPlan = originatingPlan;
        this.semantics = Objects.requireNonNull(semantics, "Program semantics must not be null");
        this.buffers = immutableCopy(buffers, "Program buffers");
        this.variables = immutableCopy(variables, "Program variables");
        this.statements = immutableCopy(statements, "Program statements");
        this.resultBuffer = Objects.requireNonNull(resultBuffer, "Program result buffer must not be null");
        this.dependenceGraph = Objects.requireNonNull(
            dependenceGraph, "Program dependence graph must not be null");

        Map<Integer, LogicalBuffer> byId = new HashMap<>();
        for (int index = 0; index < this.buffers.size(); index++) {
            LogicalBuffer buffer = this.buffers.get(index);
            if (buffer.id() != index || byId.put(buffer.id(), buffer) != null) {
                throw new IllegalArgumentException("Buffer IDs must be contiguous and deterministic");
            }
        }
        if (byId.get(resultBuffer.id()) != resultBuffer) {
            throw new IllegalArgumentException("Result buffer must belong to the program");
        }
        for (int index = 0; index < this.statements.size(); index++) {
            if (this.statements.get(index).id() != index) {
                throw new IllegalArgumentException("Statement IDs must be contiguous and deterministic");
            }
        }
        if (!dependenceGraph.statements().equals(this.statements)) {
            throw new IllegalArgumentException("Dependence graph statements must match the program");
        }
        this.buffersById = Collections.unmodifiableMap(byId);
    }

    /** True only for the exact M1 plan instance that produced this M2 program. */
    public boolean originatesFrom(ExecutionPlan plan) {
        return originatingPlan != null && originatingPlan == plan;
    }

    public OptimizationSemantics semantics() {
        return semantics;
    }

    public OptimizationSemantics optimizationSemantics() {
        return semantics;
    }

    public List<LogicalBuffer> buffers() {
        return buffers;
    }

    public List<AffineVariable> variables() {
        return variables;
    }

    public List<AffineStatement> statements() {
        return statements;
    }

    public LogicalBuffer resultBuffer() {
        return resultBuffer;
    }

    public LogicalBuffer outputBuffer() {
        return resultBuffer;
    }

    public DependenceGraph dependenceGraph() {
        return dependenceGraph;
    }

    public DependenceGraph dependences() {
        return dependenceGraph;
    }

    public LogicalBuffer buffer(int id) {
        LogicalBuffer buffer = buffersById.get(id);
        if (buffer == null) {
            throw new IllegalArgumentException("Unknown logical buffer ID: " + id);
        }
        return buffer;
    }

    public BufferLifetime lifetime(LogicalBuffer buffer) {
        if (buffer == null || buffersById.get(buffer.id()) != buffer) {
            throw new IllegalArgumentException("Buffer must belong to this program");
        }
        return buffer.lifetime();
    }

    /**
     * Deterministic human-readable dump of buffers, statements, accesses, and
     * dependence facts.
     */
    public String dump() {
        StringBuilder result = new StringBuilder();
        result.append("semantics: ").append(semantics).append('\n');
        result.append("result: %").append(resultBuffer.id()).append('\n');
        result.append("buffers:\n");
        for (LogicalBuffer buffer : buffers) {
            result.append("  %").append(buffer.id()).append(' ').append(buffer.name())
                .append(' ').append(buffer.kind().toString().toLowerCase())
                .append(' ').append(buffer.ownership().toString().toLowerCase())
                .append(' ').append(buffer.memorySpace().toString().toLowerCase())
                .append(" shape=").append(buffer.shape())
                .append(" lifetime=").append(buffer.lifetime())
                .append('\n');
        }
        result.append("statements:\n");
        if (statements.isEmpty()) {
            result.append("  (none)\n");
        } else {
            for (AffineStatement statement : statements) {
                result.append("  ").append(statement).append('\n');
                if (statement.hasReduction()) {
                    result.append("    reduction: ").append(statement.reduction()).append('\n');
                }
            }
        }
        result.append("accesses:\n");
        if (statements.stream().noneMatch(statement -> !statement.accesses().isEmpty())) {
            result.append("  (none)\n");
        } else {
            for (AffineStatement statement : statements) {
                for (AffineAccess access : statement.accesses()) {
                    result.append("  ").append(statement.name()).append(' ').append(access).append('\n');
                }
            }
        }
        result.append(dependenceGraph.dump());
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }

    private static <T> List<T> immutableCopy(List<T> source, String role) {
        if (source == null) {
            throw new IllegalArgumentException(role + " must not be null");
        }
        ArrayList<T> copy = new ArrayList<>(source);
        if (copy.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException(role + " must not contain nulls");
        }
        return Collections.unmodifiableList(copy);
    }
}
