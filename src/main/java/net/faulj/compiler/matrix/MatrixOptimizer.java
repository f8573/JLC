package net.faulj.compiler.matrix;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * M1 optimizer for immutable matrix-expression DAGs.
 */
public final class MatrixOptimizer {
    private final OptimizationSemantics semantics;
    private final MatrixCostModel costModel;

    public MatrixOptimizer() {
        this(OptimizationSemantics.STRICT, new FlopCostModel());
    }

    public MatrixOptimizer(OptimizationSemantics semantics) {
        this(semantics, new FlopCostModel());
    }

    public MatrixOptimizer(OptimizationSemantics semantics, MatrixCostModel costModel) {
        if (semantics == null) {
            throw new IllegalArgumentException("Optimization semantics must not be null");
        }
        if (costModel == null) {
            throw new IllegalArgumentException("Cost model must not be null");
        }
        this.semantics = semantics;
        this.costModel = costModel;
    }

    public OptimizationSemantics semantics() {
        return semantics;
    }

    public MatrixCostModel costModel() {
        return costModel;
    }

    /**
     * Canonicalize and optimize an expression without modifying it.
     */
    public ExecutionPlan optimize(MatrixExpr expression) {
        MatrixExpr.requireExpression(expression, "Expression");
        MatrixExpr canonical = new Canonicalizer().canonicalize(expression);

        IdentityHashMap<MatrixExpr, Integer> occurrenceCounts = new IdentityHashMap<>();
        countOccurrences(canonical, occurrenceCounts);

        Planner planner = new Planner(occurrenceCounts);
        PlanNode root = planner.plan(canonical);
        return new ExecutionPlan(root, semantics, costModel);
    }

    private static void countOccurrences(MatrixExpr expression,
                                         IdentityHashMap<MatrixExpr, Integer> counts) {
        Integer previous = counts.get(expression);
        counts.put(expression, previous == null ? 1 : previous + 1);
        if (previous != null) {
            return;
        }
        if (expression instanceof MatMul matMul) {
            countOccurrences(matMul.lhs(), counts);
            countOccurrences(matMul.rhs(), counts);
        } else if (expression instanceof Add add) {
            countOccurrences(add.lhs(), counts);
            countOccurrences(add.rhs(), counts);
        } else if (expression instanceof Scale scale) {
            countOccurrences(scale.operand(), counts);
        } else if (expression instanceof Transpose transpose) {
            countOccurrences(transpose.operand(), counts);
        }
    }

    private final class Planner {
        private final IdentityHashMap<MatrixExpr, Integer> occurrenceCounts;
        private final IdentityHashMap<MatrixExpr, PlanNode> memo = new IdentityHashMap<>();

        private Planner(IdentityHashMap<MatrixExpr, Integer> occurrenceCounts) {
            this.occurrenceCounts = occurrenceCounts;
        }

        private PlanNode plan(MatrixExpr expression) {
            PlanNode cached = memo.get(expression);
            if (cached != null) {
                return cached;
            }

            PlanNode planned;
            if (expression instanceof Input || expression instanceof SymbolicInput) {
                planned = new PlanInput(expression);
            } else if (expression instanceof MatMul matMul) {
                planned = semantics.allowsMatMulReassociation()
                    ? planMatMulChain(matMul)
                    : new PlanMatMul(plan(matMul.lhs()), plan(matMul.rhs()));
            } else if (expression instanceof Add add) {
                planned = new PlanAdd(plan(add.lhs()), plan(add.rhs()));
            } else if (expression instanceof Scale scale) {
                planned = new PlanScale(scale.factor(), plan(scale.operand()));
            } else {
                Transpose transpose = (Transpose) expression;
                planned = new PlanTranspose(plan(transpose.operand()));
            }

            memo.put(expression, planned);
            return planned;
        }

        private PlanNode planMatMulChain(MatMul root) {
            List<MatrixExpr> factors = new ArrayList<>();
            flattenChain(root, root, factors);

            List<PlanNode> plannedFactors = new ArrayList<>(factors.size());
            for (MatrixExpr factor : factors) {
                plannedFactors.add(plan(factor));
            }

            if (plannedFactors.size() == 1) {
                return plannedFactors.get(0);
            }
            if (plannedFactors.size() == 2) {
                return new PlanMatMul(plannedFactors.get(0), plannedFactors.get(1));
            }

            return new ChainPlanner(plannedFactors).build();
        }

        private void flattenChain(MatrixExpr expression,
                                  MatrixExpr chainRoot,
                                  List<MatrixExpr> factors) {
            boolean canFlatten = expression == chainRoot
                || occurrenceCounts.getOrDefault(expression, 0) == 1;
            if (expression instanceof MatMul matMul && canFlatten) {
                flattenChain(matMul.lhs(), chainRoot, factors);
                flattenChain(matMul.rhs(), chainRoot, factors);
            } else {
                factors.add(expression);
            }
        }
    }

    private final class ChainPlanner {
        private final List<PlanNode> factors;
        private final int size;
        private final CostEstimate[][] costs;
        private final int[][] splits;

        private ChainPlanner(List<PlanNode> factors) {
            this.factors = factors;
            this.size = factors.size();
            this.costs = new CostEstimate[size][size];
            this.splits = new int[size][size];
        }

        private PlanNode build() {
            for (int i = 0; i < size; i++) {
                costs[i][i] = CostEstimate.ZERO;
                splits[i][i] = -1;
            }

            for (int length = 2; length <= size; length++) {
                for (int start = 0; start + length <= size; start++) {
                    int end = start + length - 1;
                    CostEstimate best = null;
                    int bestSplit = -1;
                    for (int split = start; split < end; split++) {
                        MatrixShape leftShape = chainShape(start, split);
                        MatrixShape rightShape = chainShape(split + 1, end);
                        CostEstimate candidate = costs[start][split]
                            .plus(costs[split + 1][end])
                            .plus(costModel.estimateMatMul(leftShape, rightShape));
                        if (isBetter(candidate, best, split, bestSplit)) {
                            best = candidate;
                            bestSplit = split;
                        }
                    }
                    costs[start][end] = best;
                    splits[start][end] = bestSplit;
                }
            }
            return reconstruct(0, size - 1);
        }

        private boolean isBetter(CostEstimate candidate,
                                 CostEstimate current,
                                 int candidateSplit,
                                 int currentSplit) {
            if (current == null) {
                return true;
            }
            if (candidate.scalarMultiplications() != current.scalarMultiplications()) {
                return candidate.scalarMultiplications() < current.scalarMultiplications();
            }
            if (candidate.estimatedOutputBytes() != current.estimatedOutputBytes()) {
                return candidate.estimatedOutputBytes() < current.estimatedOutputBytes();
            }
            return candidateSplit < currentSplit;
        }

        private MatrixShape chainShape(int start, int end) {
            return new MatrixShape(factors.get(start).shape().rows(), factors.get(end).shape().columns());
        }

        private PlanNode reconstruct(int start, int end) {
            if (start == end) {
                return factors.get(start);
            }
            int split = splits[start][end];
            return new PlanMatMul(reconstruct(start, split), reconstruct(split + 1, end));
        }
    }

    private static final class Canonicalizer {
        private final IdentityHashMap<MatrixExpr, MatrixExpr> memo = new IdentityHashMap<>();

        private MatrixExpr canonicalize(MatrixExpr expression) {
            MatrixExpr cached = memo.get(expression);
            if (cached != null) {
                return cached;
            }

            MatrixExpr canonical;
            if (expression instanceof Input || expression instanceof SymbolicInput) {
                canonical = expression;
            } else if (expression instanceof MatMul matMul) {
                canonical = new MatMul(canonicalize(matMul.lhs()), canonicalize(matMul.rhs()));
            } else if (expression instanceof Add add) {
                canonical = new Add(canonicalize(add.lhs()), canonicalize(add.rhs()));
            } else if (expression instanceof Scale scale) {
                MatrixExpr operand = canonicalize(scale.operand());
                canonical = scale.factor() == 1.0 ? operand : new Scale(scale.factor(), operand);
            } else {
                Transpose transpose = (Transpose) expression;
                MatrixExpr operand = canonicalize(transpose.operand());
                canonical = operand instanceof Transpose nested
                    ? nested.operand()
                    : new Transpose(operand);
            }

            memo.put(expression, canonical);
            return canonical;
        }
    }
}
