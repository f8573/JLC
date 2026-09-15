package net.faulj.compiler.matrix.affine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.ExecutionPlan;
import net.faulj.compiler.matrix.PlanAdd;
import net.faulj.compiler.matrix.PlanInput;
import net.faulj.compiler.matrix.PlanMatMul;
import net.faulj.compiler.matrix.PlanNode;
import net.faulj.compiler.matrix.PlanScale;
import net.faulj.compiler.matrix.PlanTranspose;
import net.faulj.matrix.Matrix;

/**
 * Deterministic lowering from an M1 {@link ExecutionPlan} to M2 semantic IR.
 *
 * <p>The lowerer walks the selected plan in producer-before-consumer order and
 * memoizes plan nodes by identity. It creates logical descriptions only; it
 * never allocates, executes, fuses, or replaces the M1 runtime plan.</p>
 */
public final class AffineLowerer {
    private AffineLowerer() {
    }

    public static AffineProgram lower(ExecutionPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("Execution plan must not be null");
        }
        return new Builder(plan).build();
    }

    /**
     * Lower through the M1 plan boundary. This convenience method still
     * creates an {@code ExecutionPlan} before affine lowering.
     */
    public static AffineProgram lower(MatrixExpr expression) {
        if (expression == null) {
            throw new IllegalArgumentException("Expression must not be null");
        }
        return lower(net.faulj.compiler.matrix.MatrixCompiler.compile(expression));
    }

    private static final class Builder {
        private final ExecutionPlan plan;
        private final List<BufferDraft> buffers = new ArrayList<>();
        private final List<StatementDraft> statements = new ArrayList<>();
        private final LinkedHashMap<String, AffineVariable> variables = new LinkedHashMap<>();
        private final IdentityHashMap<PlanNode, BufferDraft> nodeBuffers = new IdentityHashMap<>();
        private final IdentityHashMap<Matrix, BufferDraft> externalBuffers = new IdentityHashMap<>();
        private final IdentityHashMap<MatrixExpr, BufferDraft> symbolicBuffers = new IdentityHashMap<>();
        private int temporaryCount;
        private int externalCount;
        private int symbolicCount;

        private Builder(ExecutionPlan plan) {
            this.plan = plan;
        }

        private AffineProgram build() {
            BufferDraft result = lowerNode(plan.root());

            IdentityHashMap<BufferDraft, LogicalBuffer> frozenBuffers = new IdentityHashMap<>();
            List<LogicalBuffer> logicalBuffers = new ArrayList<>(buffers.size());
            for (BufferDraft buffer : buffers) {
                BufferLifetime lifetime = buffer.lifetime();
                LogicalBuffer frozen = new LogicalBuffer(
                    buffer.id,
                    buffer.name,
                    buffer.kind,
                    buffer.ownership,
                    buffer.memorySpace,
                    buffer.shape,
                    lifetime);
                frozenBuffers.put(buffer, frozen);
                logicalBuffers.add(frozen);
            }

            List<AffineStatement> affineStatements = new ArrayList<>(statements.size());
            for (StatementDraft statement : statements) {
                List<AffineAccess> accesses = new ArrayList<>(statement.accesses.size());
                for (AccessDraft access : statement.accesses) {
                    accesses.add(new AffineAccess(
                        frozenBuffers.get(access.buffer), access.kind, access.indices));
                }
                affineStatements.add(new AffineStatement(
                    statement.id,
                    statement.kind,
                    statement.domain,
                    accesses,
                    statement.computation,
                    statement.reduction));
            }

            DependenceGraph dependenceGraph = DependenceGraph.analyze(affineStatements);
            return new AffineProgram(
                plan,
                plan.semantics(),
                logicalBuffers,
                new ArrayList<>(variables.values()),
                affineStatements,
                frozenBuffers.get(result),
                dependenceGraph);
        }

        private BufferDraft lowerNode(PlanNode node) {
            BufferDraft cached = nodeBuffers.get(node);
            if (cached != null) {
                return cached;
            }

            BufferDraft lowered;
            if (node instanceof PlanInput input) {
                lowered = lowerInput(input);
            } else if (node instanceof PlanAdd add) {
                lowered = lowerAdd(add);
            } else if (node instanceof PlanScale scale) {
                lowered = lowerScale(scale);
            } else if (node instanceof PlanTranspose transpose) {
                lowered = lowerTranspose(transpose);
            } else if (node instanceof PlanMatMul matMul) {
                lowered = lowerMatMul(matMul);
            } else {
                throw new IllegalArgumentException("Unsupported plan node: " + node.getClass().getName());
            }
            nodeBuffers.put(node, lowered);
            return lowered;
        }

        private BufferDraft lowerInput(PlanInput input) {
            MatrixExpr source = input.source();
            if (source instanceof net.faulj.compiler.matrix.Input runtimeInput) {
                Matrix matrix = runtimeInput.matrix();
                BufferDraft cached = externalBuffers.get(matrix);
                if (cached != null) {
                    return cached;
                }
                String name = runtimeInput.hasName() ? runtimeInput.name() : "input" + externalCount;
                externalCount++;
                BufferDraft created = newBuffer(
                    name,
                    BufferKind.EXTERNAL_INPUT,
                    BufferOwnership.BORROWED,
                    MemorySpace.from(matrix),
                    runtimeInput.shape(),
                    true);
                externalBuffers.put(matrix, created);
                return created;
            }

            net.faulj.compiler.matrix.SymbolicInput symbolicInput
                = (net.faulj.compiler.matrix.SymbolicInput) source;
            BufferDraft cached = symbolicBuffers.get(symbolicInput);
            if (cached != null) {
                return cached;
            }
            String name = symbolicInput.hasName() ? symbolicInput.name() : "symbolic" + symbolicCount;
            symbolicCount++;
            BufferDraft created = newBuffer(
                name,
                BufferKind.SYMBOLIC,
                BufferOwnership.NONE,
                MemorySpace.UNKNOWN,
                symbolicInput.shape(),
                false);
            symbolicBuffers.put(symbolicInput, created);
            return created;
        }

        private BufferDraft lowerAdd(PlanAdd add) {
            BufferDraft left = lowerNode(add.lhs());
            BufferDraft right = lowerNode(add.rhs());
            BufferDraft output = newTemporary(add.shape());
            AffineVariable i = variable("i");
            AffineVariable j = variable("j");
            AffineExpr[] outputIndices = indices(i, j);
            addStatement(
                StatementKind.ADD,
                matrixDomain(add.shape(), i, j),
                List.of(
                    access(AccessKind.WRITE, output, outputIndices),
                    access(AccessKind.READ, left, outputIndices),
                    access(AccessKind.READ, right, outputIndices)),
                location(output, outputIndices) + " = "
                    + location(left, outputIndices) + " + " + location(right, outputIndices),
                null,
                output);
            return output;
        }

        private BufferDraft lowerScale(PlanScale scale) {
            BufferDraft operand = lowerNode(scale.operand());
            BufferDraft output = newTemporary(scale.shape());
            AffineVariable i = variable("i");
            AffineVariable j = variable("j");
            AffineExpr[] indices = indices(i, j);
            addStatement(
                StatementKind.SCALE,
                matrixDomain(scale.shape(), i, j),
                List.of(
                    access(AccessKind.WRITE, output, indices),
                    access(AccessKind.READ, operand, indices)),
                location(output, indices) + " = " + Double.toString(scale.factor()) + " * "
                    + location(operand, indices),
                null,
                output);
            return output;
        }

        private BufferDraft lowerTranspose(PlanTranspose transpose) {
            BufferDraft operand = lowerNode(transpose.operand());
            BufferDraft output = newTemporary(transpose.shape());
            AffineVariable i = variable("i");
            AffineVariable j = variable("j");
            AffineExpr[] sourceIndices = indices(i, j);
            AffineExpr[] outputIndices = indices(j, i);
            addStatement(
                StatementKind.TRANSPOSE,
                matrixDomain(operand.shape, i, j),
                List.of(
                    access(AccessKind.WRITE, output, outputIndices),
                    access(AccessKind.READ, operand, sourceIndices)),
                location(output, outputIndices) + " = " + location(operand, sourceIndices),
                null,
                output);
            return output;
        }

        private BufferDraft lowerMatMul(PlanMatMul matMul) {
            BufferDraft left = lowerNode(matMul.lhs());
            BufferDraft right = lowerNode(matMul.rhs());
            BufferDraft output = newTemporary(matMul.shape());
            AffineVariable i = variable("i");
            AffineVariable j = variable("j");
            AffineVariable k = variable("k");
            AffineExpr[] outputIndices = indices(i, j);
            addStatement(
                StatementKind.MATMUL_INIT,
                matrixDomain(matMul.shape(), i, j),
                List.of(access(AccessKind.WRITE, output, outputIndices)),
                location(output, outputIndices) + " = 0",
                null,
                output);

            AffineExpr[] leftIndices = indices(i, k);
            AffineExpr[] rightIndices = indices(k, j);
            ReductionMetadata reduction = new ReductionMetadata(
                k,
                plan.semantics() == OptimizationSemantics.STRICT
                    ? ReductionSemantics.STRICT_ORDERED
                    : ReductionSemantics.REASSOCIATION_ELIGIBLE);
            addStatement(
                StatementKind.MATMUL_UPDATE,
                matMulDomain(left.shape.rows(), right.shape.columns(), left.shape.columns(), i, j, k),
                List.of(
                    access(AccessKind.READ, left, leftIndices),
                    access(AccessKind.READ, right, rightIndices),
                    access(AccessKind.REDUCTION, output, outputIndices)),
                location(output, outputIndices) + " += " + location(left, leftIndices)
                    + " * " + location(right, rightIndices),
                reduction,
                null);
            return output;
        }

        private BufferDraft newTemporary(MatrixShape shape) {
            return newBuffer(
                "tmp" + temporaryCount++,
                BufferKind.TEMPORARY,
                BufferOwnership.OWNED,
                MemorySpace.UNKNOWN,
                shape,
                false);
        }

        private BufferDraft newBuffer(String name,
                                      BufferKind kind,
                                      BufferOwnership ownership,
                                      MemorySpace memorySpace,
                                      MatrixShape shape,
                                      boolean liveForEvaluation) {
            BufferDraft created = new BufferDraft(
                buffers.size(), name, kind, ownership, memorySpace, shape, liveForEvaluation);
            buffers.add(created);
            return created;
        }

        private void addStatement(StatementKind kind,
                                  IterationDomain domain,
                                  List<AccessDraft> accesses,
                                  String computation,
                                  ReductionMetadata reduction,
                                  BufferDraft producer) {
            int id = statements.size();
            for (AccessDraft access : accesses) {
                access.buffer.observeAccess(id, access.kind);
            }
            if (producer != null) {
                producer.producerStatementId = id;
            }
            statements.add(new StatementDraft(
                id,
                kind,
                domain,
                new ArrayList<>(accesses),
                computation,
                reduction));
        }

        private AffineVariable variable(String name) {
            AffineVariable variable = variables.get(name);
            if (variable != null) {
                return variable;
            }
            AffineVariable created = new AffineVariable(name);
            variables.put(name, created);
            return created;
        }

        private static AffineExpr[] indices(AffineVariable first, AffineVariable second) {
            return new AffineExpr[]{AffineExpr.variable(first), AffineExpr.variable(second)};
        }

        private static IterationDomain matrixDomain(MatrixShape shape,
                                                    AffineVariable i,
                                                    AffineVariable j) {
            return IterationDomain.of(
                IterationDomain.range(i, 0L, shape.rows()),
                IterationDomain.range(j, 0L, shape.columns()));
        }

        private static IterationDomain matMulDomain(int rows,
                                                     int columns,
                                                     int reduction,
                                                     AffineVariable i,
                                                     AffineVariable j,
                                                     AffineVariable k) {
            return IterationDomain.of(
                IterationDomain.range(i, 0L, rows),
                IterationDomain.range(j, 0L, columns),
                IterationDomain.range(k, 0L, reduction));
        }

        private static AccessDraft access(AccessKind kind,
                                          BufferDraft buffer,
                                          AffineExpr[] indices) {
            return new AccessDraft(kind, buffer, List.of(indices));
        }

        private static String location(BufferDraft buffer, AffineExpr[] indices) {
            return "%" + buffer.id + "[" + String.join(",", toStrings(indices)) + "]";
        }

        private static List<String> toStrings(AffineExpr[] expressions) {
            List<String> result = new ArrayList<>(expressions.length);
            for (AffineExpr expression : expressions) {
                result.add(expression.toString());
            }
            return result;
        }
    }

    private static final class BufferDraft {
        private final int id;
        private final String name;
        private final BufferKind kind;
        private final BufferOwnership ownership;
        private final MemorySpace memorySpace;
        private final MatrixShape shape;
        private final boolean liveForEvaluation;
        private Integer producerStatementId;
        private Integer firstUseStatementId;
        private Integer lastUseStatementId;

        private BufferDraft(int id,
                            String name,
                            BufferKind kind,
                            BufferOwnership ownership,
                            MemorySpace memorySpace,
                            MatrixShape shape,
                            boolean liveForEvaluation) {
            this.id = id;
            this.name = name;
            this.kind = kind;
            this.ownership = ownership;
            this.memorySpace = memorySpace;
            this.shape = shape;
            this.liveForEvaluation = liveForEvaluation;
        }

        private void observeAccess(int statementId, AccessKind kind) {
            if (!kind.reads()) {
                return;
            }
            if (firstUseStatementId == null) {
                firstUseStatementId = statementId;
            }
            lastUseStatementId = statementId;
        }

        private BufferLifetime lifetime() {
            if (kind == BufferKind.SYMBOLIC) {
                return BufferLifetime.none();
            }
            return new BufferLifetime(
                producerStatementId,
                firstUseStatementId,
                lastUseStatementId,
                liveForEvaluation);
        }
    }

    private static final class AccessDraft {
        private final AccessKind kind;
        private final BufferDraft buffer;
        private final List<AffineExpr> indices;

        private AccessDraft(AccessKind kind, BufferDraft buffer, List<AffineExpr> indices) {
            this.kind = kind;
            this.buffer = buffer;
            this.indices = Collections.unmodifiableList(new ArrayList<>(indices));
        }
    }

    private static final class StatementDraft {
        private final int id;
        private final StatementKind kind;
        private final IterationDomain domain;
        private final List<AccessDraft> accesses;
        private final String computation;
        private final ReductionMetadata reduction;

        private StatementDraft(int id,
                               StatementKind kind,
                               IterationDomain domain,
                               List<AccessDraft> accesses,
                               String computation,
                               ReductionMetadata reduction) {
            this.id = id;
            this.kind = kind;
            this.domain = domain;
            this.accesses = Collections.unmodifiableList(new ArrayList<>(accesses));
            this.computation = computation;
            this.reduction = reduction;
        }
    }
}
