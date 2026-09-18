package net.faulj.compiler.matrix.kernel;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.matrix.Matrix;

/**
 * Straightforward scalar reference executor for verified Kernel IR.
 *
 * <p>Compilation of affine expressions and the scalar scratch arrays happens
 * once per invocation. The point loop allocates no maps, lists, or scalar
 * objects. This class is a correctness backend, not a performance backend.</p>
 */
public final class KernelReferenceExecutor {
    private KernelReferenceExecutor() {
    }

    /** Verify before execution. Prefer {@link #executeVerified} for a cached result. */
    public static Matrix execute(KernelProgram program, KernelBinding binding) {
        KernelVerifier.requireValid(program);
        return executeVerified(program, binding);
    }

    public static Matrix execute(KernelFunction function, KernelBinding binding) {
        KernelVerifier.requireValid(function);
        return executeVerified(function, binding);
    }

    /** Convenience form that allocates a heap output only when no output is bound. */
    public static Matrix execute(KernelProgram program,
                                 Map<KernelBuffer, Matrix> matrices) {
        if (program == null || matrices == null) {
            throw new IllegalArgumentException("Kernel program and matrices are required");
        }
        KernelFunction function = program.function();
        IdentityHashMap<KernelBuffer, Matrix> resolved = new IdentityHashMap<>(matrices);
        if (function.outputBuffers().size() == 1) {
            KernelBuffer output = function.outputBuffers().get(0);
            if (!resolved.containsKey(output) || resolved.get(output) == null) {
                resolved.put(output, new Matrix(output.shape().rows(), output.shape().columns()));
            }
        }
        return execute(program, new KernelBinding(function, resolved));
    }

    /** Execute IR whose caller has already retained a valid verifier result. */
    public static Matrix executeVerified(KernelProgram program, KernelBinding binding) {
        if (program == null) {
            throw new IllegalArgumentException("Kernel program must not be null");
        }
        if (program.functions().size() != 1) {
            throw new KernelExecutionException(
                "R3 reference executor supports one kernel function, got "
                    + program.functions().size());
        }
        return executeVerified(program.function(), binding);
    }

    /** Execute a function whose caller has already retained a valid verifier result. */
    public static Matrix executeVerified(KernelFunction function, KernelBinding binding) {
        if (function == null || binding == null) {
            throw new IllegalArgumentException("Kernel function and binding are required");
        }
        if (binding.function() != function) {
            throw new KernelExecutionException("Kernel binding belongs to a different function");
        }
        IdentityHashMap<KernelBuffer, Matrix> matrices = resolveMatrices(function, binding);
        CompiledFunction compiled = CompiledFunction.compile(function);
        compiled.execute(matrices);
        return matrices.get(function.outputBuffers().get(0));
    }

    private static IdentityHashMap<KernelBuffer, Matrix> resolveMatrices(
        KernelFunction function,
        KernelBinding binding) {
        IdentityHashMap<KernelBuffer, Matrix> result = new IdentityHashMap<>();
        for (KernelBuffer buffer : function.buffers()) {
            Matrix matrix = binding.matrix(buffer);
            if (matrix == null) {
                throw new KernelExecutionException(
                    "Kernel buffer %" + buffer.id() + " is unresolved");
            }
            if (matrix.getRowCount() != buffer.shape().rows()
                || matrix.getColumnCount() != buffer.shape().columns()) {
                throw new KernelExecutionException(
                    "Kernel buffer %" + buffer.id() + " shape "
                        + matrix.getRowCount() + "x" + matrix.getColumnCount()
                        + " does not match " + buffer.shape());
            }
            if (matrix.hasImagData()) {
                throw new KernelExecutionException(
                    "complex FP64 storage is unsupported by the R3 reference executor");
            }
            result.put(buffer, matrix);
        }
        return result;
    }

    private static final class CompiledFunction {
        private final VariableLayout variables;
        private final CompiledLoop[] loops;
        private final CompiledOperation[] operations;
        private final int valueArraySize;

        private CompiledFunction(VariableLayout variables,
                                 CompiledLoop[] loops,
                                 CompiledOperation[] operations,
                                 int valueArraySize) {
            this.variables = variables;
            this.loops = loops;
            this.operations = operations;
            this.valueArraySize = valueArraySize;
        }

        private static CompiledFunction compile(KernelFunction function) {
            VariableLayout variables = new VariableLayout();
            CompiledLoop[] loops = new CompiledLoop[function.loops().size()];
            for (int index = 0; index < loops.length; index++) {
                KernelLoop loop = function.loops().get(index);
                variables.add(loop.inductionVariable());
                variables.add(loop.semanticVariable());
                if (loop.valueExpression() != null) {
                    variables.addAll(loop.valueExpression());
                }
                loops[index] = new CompiledLoop(
                    variables.slot(loop.inductionVariable()),
                    variables.slot(loop.semanticVariable()), loop.lowerBound(),
                    loop.upperBound(), loop.step(), loop.valueExpression() == null
                        ? null : CompiledExpression.compile(loop.valueExpression(), variables));
            }
            CompiledOperation[] operations = new CompiledOperation[function.body().operations().size()];
            int maxValueId = -1;
            for (KernelValue value : function.values()) {
                maxValueId = Math.max(maxValueId, value.id());
            }
            for (int index = 0; index < operations.length; index++) {
                KernelOp operation = function.body().operations().get(index);
                if (operation.access() != null) {
                    variables.addAll(operation.access().indices());
                }
            }
            for (int index = 0; index < operations.length; index++) {
                KernelOp operation = function.body().operations().get(index);
                CompiledAccess access = operation.access() == null
                    ? null : CompiledAccess.compile(operation.access(), variables);
                operations[index] = new CompiledOperation(
                    operation.opcode(), operation.resultValueId(), operation.operands(),
                    operation.immediate(), access);
            }
            return new CompiledFunction(variables, loops, operations, maxValueId + 1);
        }

        private void execute(IdentityHashMap<KernelBuffer, Matrix> matrices) {
            long[] bindings = new long[variables.size()];
            boolean[] bound = new boolean[variables.size()];
            double[] values = new double[valueArraySize];
            visit(0, bindings, bound, values, matrices);
        }

        private void visit(int loopIndex,
                           long[] bindings,
                           boolean[] bound,
                           double[] values,
                           IdentityHashMap<KernelBuffer, Matrix> matrices) {
            if (loopIndex == loops.length) {
                executePoint(bindings, bound, values, matrices);
                return;
            }
            CompiledLoop loop = loops[loopIndex];
            boolean oldInductionBound = bound[loop.inductionSlot];
            long oldInduction = bindings[loop.inductionSlot];
            boolean sameSlot = loop.inductionSlot == loop.semanticSlot;
            boolean oldSemanticBound = sameSlot
                ? oldInductionBound : bound[loop.semanticSlot];
            long oldSemantic = sameSlot ? oldInduction : bindings[loop.semanticSlot];
            for (long induction = loop.lowerBound;
                 induction < loop.upperBound;
                 induction += loop.step) {
                bindings[loop.inductionSlot] = induction;
                bound[loop.inductionSlot] = true;
                if (loop.valueExpression == null) {
                    bound[loop.semanticSlot] = false;
                } else {
                    bindings[loop.semanticSlot] = loop.valueExpression.evaluate(bindings, bound);
                    bound[loop.semanticSlot] = true;
                }
                visit(loopIndex + 1, bindings, bound, values, matrices);
                if (induction > Long.MAX_VALUE - loop.step) {
                    break;
                }
            }
            if (sameSlot) {
                bindings[loop.inductionSlot] = oldInduction;
                bound[loop.inductionSlot] = oldInductionBound;
            } else {
                bindings[loop.inductionSlot] = oldInduction;
                bound[loop.inductionSlot] = oldInductionBound;
                bindings[loop.semanticSlot] = oldSemantic;
                bound[loop.semanticSlot] = oldSemanticBound;
            }
        }

        private void executePoint(long[] bindings,
                                  boolean[] bound,
                                  double[] values,
                                  IdentityHashMap<KernelBuffer, Matrix> matrices) {
            for (CompiledOperation operation : operations) {
                switch (operation.opcode) {
                    case LOAD -> {
                        Matrix matrix = matrices.get(operation.access.buffer);
                        int row = operation.access.index(0, bindings, bound);
                        int column = operation.access.index(1, bindings, bound);
                        values[operation.resultValueId] = matrix.get(row, column);
                    }
                    case CONSTANT -> values[operation.resultValueId] = operation.immediate;
                    case ADD -> values[operation.resultValueId] =
                        values[operation.operands.get(0)] + values[operation.operands.get(1)];
                    case MUL -> values[operation.resultValueId] =
                        values[operation.operands.get(0)] * values[operation.operands.get(1)];
                    case STORE -> {
                        Matrix matrix = matrices.get(operation.access.buffer);
                        int row = operation.access.index(0, bindings, bound);
                        int column = operation.access.index(1, bindings, bound);
                        matrix.set(row, column, values[operation.operands.get(0)]);
                    }
                }
            }
        }
    }

    private static final class CompiledOperation {
        private final KernelOpcode opcode;
        private final int resultValueId;
        private final java.util.List<Integer> operands;
        private final double immediate;
        private final CompiledAccess access;

        private CompiledOperation(KernelOpcode opcode,
                                  int resultValueId,
                                  java.util.List<Integer> operands,
                                  Double immediate,
                                  CompiledAccess access) {
            this.opcode = opcode;
            this.resultValueId = resultValueId;
            this.operands = operands;
            this.immediate = immediate == null ? Double.NaN : immediate;
            this.access = access;
        }
    }

    private static final class CompiledAccess {
        private final KernelBuffer buffer;
        private final CompiledExpression[] indices;

        private CompiledAccess(KernelBuffer buffer, CompiledExpression[] indices) {
            this.buffer = buffer;
            this.indices = indices;
        }

        private static CompiledAccess compile(KernelAccess access, VariableLayout variables) {
            CompiledExpression[] indices = new CompiledExpression[access.indices().size()];
            for (int index = 0; index < indices.length; index++) {
                indices[index] = CompiledExpression.compile(access.index(index), variables);
            }
            return new CompiledAccess(access.buffer(), indices);
        }

        private int index(int dimension, long[] bindings, boolean[] bound) {
            long value = indices[dimension].evaluate(bindings, bound);
            if (value < 0L || value > Integer.MAX_VALUE) {
                throw new KernelExecutionException("Kernel affine index is out of range: " + value);
            }
            return (int) value;
        }
    }

    private static final class CompiledLoop {
        private final int inductionSlot;
        private final int semanticSlot;
        private final long lowerBound;
        private final long upperBound;
        private final long step;
        private final CompiledExpression valueExpression;

        private CompiledLoop(int inductionSlot,
                             int semanticSlot,
                             long lowerBound,
                             long upperBound,
                             long step,
                             CompiledExpression valueExpression) {
            this.inductionSlot = inductionSlot;
            this.semanticSlot = semanticSlot;
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
            this.step = step;
            this.valueExpression = valueExpression;
        }
    }

    private static final class CompiledExpression {
        private final long constant;
        private final int[] slots;
        private final long[] coefficients;

        private CompiledExpression(long constant, int[] slots, long[] coefficients) {
            this.constant = constant;
            this.slots = slots;
            this.coefficients = coefficients;
        }

        private static CompiledExpression compile(AffineExpr expression,
                                                  VariableLayout variables) {
            int[] slots = new int[expression.coefficients().size()];
            long[] coefficients = new long[slots.length];
            int index = 0;
            for (Map.Entry<AffineVariable, Long> term : expression.coefficients().entrySet()) {
                slots[index] = variables.slot(term.getKey());
                coefficients[index] = term.getValue();
                index++;
            }
            return new CompiledExpression(expression.constant(), slots, coefficients);
        }

        private long evaluate(long[] bindings, boolean[] bound) {
            long result = constant;
            for (int index = 0; index < slots.length; index++) {
                int slot = slots[index];
                if (!bound[slot]) {
                    throw new KernelExecutionException(
                        "Kernel affine expression uses an unbound loop variable");
                }
                try {
                    result = Math.addExact(result,
                        Math.multiplyExact(coefficients[index], bindings[slot]));
                } catch (ArithmeticException overflow) {
                    throw new KernelExecutionException("Kernel affine expression overflow");
                }
            }
            return result;
        }
    }

    private static final class VariableLayout {
        private final Map<AffineVariable, Integer> slots = new HashMap<>();

        private void add(AffineVariable variable) {
            if (variable != null) {
                slots.computeIfAbsent(variable, ignored -> slots.size());
            }
        }

        private void addAll(AffineExpr expression) {
            for (AffineVariable variable : expression.coefficients().keySet()) {
                add(variable);
            }
        }

        private void addAll(java.util.List<AffineExpr> expressions) {
            for (AffineExpr expression : expressions) {
                addAll(expression);
            }
        }

        private int slot(AffineVariable variable) {
            Integer result = slots.get(variable);
            if (result == null) {
                throw new KernelExecutionException("Kernel variable was not compiled: " + variable);
            }
            return result;
        }

        private int size() {
            return slots.size();
        }
    }
}
