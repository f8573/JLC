package net.faulj.compiler.matrix.cpu;

import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.matrix.Matrix;

/**
 * Executable realization of one generalized R2 fused region.
 *
 * <p>The immutable {@link FusedRegionPlan} owns the semantic scalar graph.
 * This step compiles its affine indices once, then reuses primitive arrays for
 * every point. The dense real case has a separate raw-array loop; the generic
 * case still avoids the old per-point maps and index arrays.</p>
 */
public final class CpuFusedRegionStep implements CpuStep {
    public enum ExecutionPath {
        FAST_REAL_DENSE,
        GENERIC_SCHEDULE
    }

    private final int id;
    private final FusedRegionPlan regionPlan;
    private final List<LogicalBuffer> inputBuffers;
    private final CompiledProgram compiledProgram;
    private final FastProgram fastProgram;
    private volatile ExecutionPath lastExecutionPath;

    CpuFusedRegionStep(int id, FusedRegionPlan regionPlan) {
        if (id < 0) {
            throw new IllegalArgumentException("CPU step ID must be non-negative");
        }
        this.id = id;
        this.regionPlan = Objects.requireNonNull(regionPlan, "Fused region must not be null");
        this.inputBuffers = List.copyOf(regionPlan.leafBuffers());
        this.compiledProgram = CompiledProgram.compile(regionPlan);
        this.fastProgram = FastProgram.tryCompile(regionPlan);
    }

    @Override
    public int id() {
        return id;
    }

    @Override
    public CpuStepKind kind() {
        // Preserve the M4 enum contract; regionPlan() identifies this as R2.
        return CpuStepKind.FUSED_ELEMENTWISE;
    }

    @Override
    public LogicalBuffer outputBuffer() {
        return regionPlan.outputBuffer();
    }

    @Override
    public List<LogicalBuffer> inputBuffers() {
        return inputBuffers;
    }

    public FusedRegionPlan regionPlan() {
        return regionPlan;
    }

    public ScalarFusionProgram scalarProgram() {
        return regionPlan.scalarProgram();
    }

    public ScheduleBand scheduleBand() {
        return regionPlan.scheduleBand();
    }

    public List<LogicalBuffer> leafBuffers() {
        return inputBuffers;
    }

    public List<LogicalBuffer> eliminatedBuffers() {
        return regionPlan.eliminatedBuffers();
    }

    public ExecutionPath lastExecutionPath() {
        return lastExecutionPath;
    }

    @Override
    public String description() {
        return "FUSED_REGION F" + regionPlan.regionId() + " S"
            + regionPlan.statementIds() + " -> %" + outputBuffer().id()
            + " (elides " + regionPlan.eliminatedBuffers().stream()
                .map(buffer -> "%" + buffer.id()).toList() + ")";
    }

    void execute(CpuExecutionContext context) {
        Matrix output = context.allocate(outputBuffer());
        Matrix[] leaves = new Matrix[inputBuffers.size()];
        boolean allRealDense = output.getClass() == Matrix.class
            && outputBuffer().ownership() == BufferOwnership.OWNED
            && !output.hasImagData();
        for (int index = 0; index < leaves.length; index++) {
            Matrix leaf = context.value(inputBuffers.get(index));
            leaves[index] = leaf;
            allRealDense &= leaf.getClass() == Matrix.class && !leaf.hasImagData();
            if (output == leaf) {
                allRealDense = false;
            }
        }

        if (allRealDense && fastProgram != null) {
            fastProgram.execute(output, leaves);
            lastExecutionPath = ExecutionPath.FAST_REAL_DENSE;
            return;
        }

        lastExecutionPath = ExecutionPath.GENERIC_SCHEDULE;
        boolean complex = false;
        for (Matrix leaf : leaves) {
            complex |= leaf.hasImagData();
        }
        if (complex) {
            output.ensureImagData();
        }
        compiledProgram.execute(output, leaves, complex);
    }

    private static final class CompiledProgram {
        private final VariableLayout layout;
        private final CompiledLoop[] loops;
        private final CompiledNode[] nodes;
        private final CompiledIndex outputRow;
        private final CompiledIndex outputColumn;
        private final int rootNodeId;

        private CompiledProgram(VariableLayout layout,
                                CompiledLoop[] loops,
                                CompiledNode[] nodes,
                                CompiledIndex outputRow,
                                CompiledIndex outputColumn,
                                int rootNodeId) {
            this.layout = layout;
            this.loops = loops;
            this.nodes = nodes;
            this.outputRow = outputRow;
            this.outputColumn = outputColumn;
            this.rootNodeId = rootNodeId;
        }

        private static CompiledProgram compile(FusedRegionPlan region) {
            ScalarFusionProgram scalar = region.scalarProgram();
            VariableLayout layout = new VariableLayout();
            ScheduleBand band = region.scheduleBand();
            for (ScheduleLoop loop : band.loops()) {
                layout.add(loop.inductionVariable());
                layout.add(loop.semanticVariable());
                if (loop.valueExpression() != null) {
                    layout.addAll(loop.valueExpression());
                }
            }
            for (ScalarFusionProgram.ScalarNode node : scalar.nodes()) {
                for (AffineExpr index : node.indices()) {
                    layout.addAll(index);
                }
            }
            AffineAccess output = region.outputAccess();
            for (AffineExpr index : output.indices()) {
                layout.addAll(index);
            }

            CompiledLoop[] loops = new CompiledLoop[band.loops().size()];
            for (int index = 0; index < loops.length; index++) {
                ScheduleLoop loop = band.loop(index);
                loops[index] = new CompiledLoop(
                    layout.slot(loop.inductionVariable()),
                    layout.slot(loop.semanticVariable()),
                    loop.lowerBound(),
                    loop.upperBound(),
                    loop.step(),
                    loop.valueExpression() == null
                        ? null : CompiledIndex.compile(loop.valueExpression(), layout));
            }

            IdentityHashMap<LogicalBuffer, Integer> leafIndices = new IdentityHashMap<>();
            List<LogicalBuffer> leaves = region.leafBuffers();
            for (int index = 0; index < leaves.size(); index++) {
                leafIndices.put(leaves.get(index), index);
            }
            CompiledNode[] nodes = new CompiledNode[scalar.nodeCount()];
            for (ScalarFusionProgram.ScalarNode node : scalar.nodes()) {
                int firstInput = node.inputs().isEmpty() ? -1 : node.inputs().get(0);
                int secondInput = node.inputs().size() < 2 ? -1 : node.inputs().get(1);
                int leafIndex = -1;
                CompiledIndex row = null;
                CompiledIndex column = null;
                if (node.isLoad()) {
                    Integer found = leafIndices.get(node.buffer());
                    if (found == null) {
                        throw new IllegalArgumentException(
                            "Fused scalar load does not name a leaf buffer %"
                                + node.buffer().id());
                    }
                    leafIndex = found;
                    row = CompiledIndex.compile(node.indices().get(0), layout);
                    column = CompiledIndex.compile(node.indices().get(1), layout);
                }
                nodes[node.id()] = new CompiledNode(
                    node.id(), node.opcode(), firstInput, secondInput, leafIndex,
                    node.factor(), row, column);
            }
            if (output.indices().size() != 2) {
                throw new IllegalArgumentException("Fused output must have two indices");
            }
            return new CompiledProgram(
                layout, loops, nodes,
                CompiledIndex.compile(output.index(0), layout),
                CompiledIndex.compile(output.index(1), layout),
                scalar.rootNodeId());
        }

        private void execute(Matrix output, Matrix[] leaves, boolean complex) {
            long[] bindings = new long[layout.size()];
            boolean[] bound = new boolean[layout.size()];
            double[] real = new double[nodes.length];
            double[] imaginary = complex ? new double[nodes.length] : null;
            boolean[] nodeComplex = new boolean[nodes.length];
            boolean[] valid = new boolean[nodes.length];
            visit(0, bindings, bound, output, leaves, real, imaginary, nodeComplex, valid, complex);
        }

        private void visit(int loopIndex,
                           long[] bindings,
                           boolean[] bound,
                           Matrix output,
                           Matrix[] leaves,
                           double[] real,
                           double[] imaginary,
                           boolean[] nodeComplex,
                           boolean[] valid,
                           boolean complex) {
            if (loopIndex == loops.length) {
                executePoint(bindings, bound, output, leaves, real, imaginary,
                    nodeComplex, valid, complex);
                return;
            }
            CompiledLoop loop = loops[loopIndex];
            int inductionSlot = loop.inductionSlot;
            int semanticSlot = loop.semanticSlot;
            boolean oldInductionBound = bound[inductionSlot];
            long oldInduction = bindings[inductionSlot];
            boolean sameSlot = inductionSlot == semanticSlot;
            boolean oldSemanticBound = sameSlot ? oldInductionBound : bound[semanticSlot];
            long oldSemantic = sameSlot ? oldInduction : bindings[semanticSlot];
            for (long induction = loop.lowerBound;
                 induction < loop.upperBound;
                 induction += loop.step) {
                bindings[inductionSlot] = induction;
                bound[inductionSlot] = true;
                if (loop.valueExpression == null) {
                    bound[semanticSlot] = false;
                } else {
                    bindings[semanticSlot] = loop.valueExpression.evaluate(bindings, bound);
                    bound[semanticSlot] = true;
                }
                visit(loopIndex + 1, bindings, bound, output, leaves,
                    real, imaginary, nodeComplex, valid, complex);
                if (induction > Long.MAX_VALUE - loop.step) {
                    break;
                }
            }
            if (sameSlot) {
                bindings[inductionSlot] = oldInduction;
                bound[inductionSlot] = oldInductionBound;
            } else {
                bindings[inductionSlot] = oldInduction;
                bound[inductionSlot] = oldInductionBound;
                bindings[semanticSlot] = oldSemantic;
                bound[semanticSlot] = oldSemanticBound;
            }
        }

        private void executePoint(long[] bindings,
                                  boolean[] bound,
                                  Matrix output,
                                  Matrix[] leaves,
                                  double[] real,
                                  double[] imaginary,
                                  boolean[] nodeComplex,
                                  boolean[] valid,
                                  boolean complex) {
            for (CompiledNode node : nodes) {
                int nodeId = node.id;
                switch (node.opcode) {
                    case LOAD -> {
                        Matrix input = leaves[node.leafIndex];
                        int row = checkedIndex(node.row.evaluate(bindings, bound));
                        int column = checkedIndex(node.column.evaluate(bindings, bound));
                        if (!inBounds(input, row, column)) {
                            valid[nodeId] = false;
                            nodeComplex[nodeId] = false;
                        } else {
                            real[nodeId] = input.get(row, column);
                            boolean inputComplex = input.hasImagData();
                            nodeComplex[nodeId] = inputComplex;
                            if (inputComplex) {
                                imaginary[nodeId] = input.getImag(row, column);
                            }
                            valid[nodeId] = true;
                        }
                    }
                    case SCALE -> {
                        int input = node.firstInput;
                        if (!valid[input]) {
                            valid[nodeId] = false;
                            nodeComplex[nodeId] = false;
                        } else {
                            real[nodeId] = node.factor * real[input];
                            boolean inputComplex = nodeComplex[input];
                            nodeComplex[nodeId] = inputComplex;
                            if (inputComplex) {
                                imaginary[nodeId] = node.factor * imaginary[input];
                            }
                            valid[nodeId] = true;
                        }
                    }
                    case ADD -> {
                        int first = node.firstInput;
                        int second = node.secondInput;
                        if (!valid[first] || !valid[second]) {
                            valid[nodeId] = false;
                            nodeComplex[nodeId] = false;
                        } else {
                            real[nodeId] = real[first] + real[second];
                            boolean firstComplex = nodeComplex[first];
                            boolean secondComplex = nodeComplex[second];
                            nodeComplex[nodeId] = firstComplex || secondComplex;
                            if (firstComplex && secondComplex) {
                                imaginary[nodeId] = imaginary[first] + imaginary[second];
                            } else if (firstComplex) {
                                imaginary[nodeId] = imaginary[first];
                            } else if (secondComplex) {
                                imaginary[nodeId] = imaginary[second];
                            }
                            valid[nodeId] = true;
                        }
                    }
                    case TRANSPOSE -> {
                        int input = node.firstInput;
                        if (!valid[input]) {
                            valid[nodeId] = false;
                            nodeComplex[nodeId] = false;
                        } else {
                            real[nodeId] = real[input];
                            nodeComplex[nodeId] = nodeComplex[input];
                            if (nodeComplex[input]) {
                                imaginary[nodeId] = imaginary[input];
                            }
                            valid[nodeId] = true;
                        }
                    }
                }
            }

            if (!valid[rootNodeId]) {
                return;
            }
            int row = checkedIndex(outputRow.evaluate(bindings, bound));
            int column = checkedIndex(outputColumn.evaluate(bindings, bound));
            if (!inBounds(output, row, column)) {
                return;
            }
            if (complex) {
                output.setComplex(row, column, real[rootNodeId], imaginary[rootNodeId]);
            } else {
                output.set(row, column, real[rootNodeId]);
            }
        }
    }

    private static final class VariableLayout {
        private final Map<AffineVariable, Integer> slots = new HashMap<>();

        private void add(AffineVariable variable) {
            slots.computeIfAbsent(variable, ignored -> slots.size());
        }

        private void addAll(AffineExpr expression) {
            for (AffineVariable variable : expression.coefficients().keySet()) {
                add(variable);
            }
        }

        private int slot(AffineVariable variable) {
            Integer result = slots.get(variable);
            if (result == null) {
                throw new IllegalArgumentException("Uncompiled affine variable " + variable);
            }
            return result;
        }

        private int size() {
            return slots.size();
        }
    }

    private static final class CompiledLoop {
        private final int inductionSlot;
        private final int semanticSlot;
        private final long lowerBound;
        private final long upperBound;
        private final long step;
        private final CompiledIndex valueExpression;

        private CompiledLoop(int inductionSlot,
                             int semanticSlot,
                             long lowerBound,
                             long upperBound,
                             long step,
                             CompiledIndex valueExpression) {
            this.inductionSlot = inductionSlot;
            this.semanticSlot = semanticSlot;
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
            this.step = step;
            this.valueExpression = valueExpression;
        }
    }

    private static final class CompiledNode {
        private final int id;
        private final ScalarFusionProgram.Opcode opcode;
        private final int firstInput;
        private final int secondInput;
        private final int leafIndex;
        private final double factor;
        private final CompiledIndex row;
        private final CompiledIndex column;

        private CompiledNode(int id,
                             ScalarFusionProgram.Opcode opcode,
                             int firstInput,
                             int secondInput,
                             int leafIndex,
                             double factor,
                             CompiledIndex row,
                             CompiledIndex column) {
            this.id = id;
            this.opcode = opcode;
            this.firstInput = firstInput;
            this.secondInput = secondInput;
            this.leafIndex = leafIndex;
            this.factor = factor;
            this.row = row;
            this.column = column;
        }
    }

    private static final class CompiledIndex {
        private final long constant;
        private final int[] slots;
        private final long[] coefficients;

        private CompiledIndex(long constant, int[] slots, long[] coefficients) {
            this.constant = constant;
            this.slots = slots;
            this.coefficients = coefficients;
        }

        private static CompiledIndex compile(AffineExpr expression, VariableLayout layout) {
            int[] slots = new int[expression.coefficients().size()];
            long[] coefficients = new long[slots.length];
            int index = 0;
            for (Map.Entry<AffineVariable, Long> term : expression.coefficients().entrySet()) {
                slots[index] = layout.slot(term.getKey());
                coefficients[index] = term.getValue();
                index++;
            }
            return new CompiledIndex(expression.constant(), slots, coefficients);
        }

        private long evaluate(long[] bindings, boolean[] bound) {
            long result = constant;
            for (int index = 0; index < slots.length; index++) {
                int slot = slots[index];
                if (!bound[slot]) {
                    throw new IllegalStateException(
                        "Schedule does not bind affine variable slot " + slot);
                }
                result = Math.addExact(result,
                    Math.multiplyExact(coefficients[index], bindings[slot]));
            }
            return result;
        }
    }

    private static final class FastProgram {
        private final FastLoop firstLoop;
        private final FastLoop secondLoop;
        private final FastNode[] nodes;
        private final FastIndex outputRow;
        private final FastIndex outputColumn;
        private final int rootNodeId;

        private FastProgram(FastLoop firstLoop,
                            FastLoop secondLoop,
                            FastNode[] nodes,
                            FastIndex outputRow,
                            FastIndex outputColumn,
                            int rootNodeId) {
            this.firstLoop = firstLoop;
            this.secondLoop = secondLoop;
            this.nodes = nodes;
            this.outputRow = outputRow;
            this.outputColumn = outputColumn;
            this.rootNodeId = rootNodeId;
        }

        private static FastProgram tryCompile(FusedRegionPlan region) {
            ScheduleBand band = region.scheduleBand();
            if (band.loops().size() != 2 || region.iterationDomain().ranges().size() != 2) {
                return null;
            }
            Map<AffineVariable, Integer> semanticLoops = new HashMap<>();
            FastLoop[] loops = new FastLoop[2];
            for (int index = 0; index < 2; index++) {
                ScheduleLoop loop = band.loop(index);
                if (!loop.annotations().isEmpty() || loop.guard() != null
                    || loop.valueExpression() == null
                    || !loop.valueExpression().equals(
                        AffineExpr.variable(loop.inductionVariable()))
                    || loop.lowerBound() < 0L || loop.step() != 1L
                    || semanticLoops.put(loop.semanticVariable(), index) != null) {
                    return null;
                }
                long extent = region.iterationDomain().extent(loop.semanticVariable());
                if (extent < 0L || loop.lowerBound() != 0L || loop.upperBound() != extent) {
                    return null;
                }
                loops[index] = new FastLoop(loop.lowerBound(), loop.upperBound());
            }
            if (semanticLoops.size() != 2) {
                return null;
            }
            FastIndex outputRow = FastIndex.compile(
                region.outputAccess().index(0), semanticLoops);
            FastIndex outputColumn = FastIndex.compile(
                region.outputAccess().index(1), semanticLoops);
            if (outputRow == null || outputColumn == null) {
                return null;
            }

            List<LogicalBuffer> leaves = region.leafBuffers();
            IdentityHashMap<LogicalBuffer, Integer> leafIndices = new IdentityHashMap<>();
            for (int index = 0; index < leaves.size(); index++) {
                leafIndices.put(leaves.get(index), index);
            }
            FastNode[] nodes = new FastNode[region.scalarProgram().nodeCount()];
            for (ScalarFusionProgram.ScalarNode node : region.scalarProgram().nodes()) {
                FastIndex row = null;
                FastIndex column = null;
                int leafIndex = -1;
                if (node.isLoad()) {
                    row = FastIndex.compile(node.indices().get(0), semanticLoops);
                    column = FastIndex.compile(node.indices().get(1), semanticLoops);
                    Integer found = leafIndices.get(node.buffer());
                    if (row == null || column == null || found == null) {
                        return null;
                    }
                    leafIndex = found;
                }
                int first = node.inputs().isEmpty() ? -1 : node.inputs().get(0);
                int second = node.inputs().size() < 2 ? -1 : node.inputs().get(1);
                nodes[node.id()] = new FastNode(
                    node.id(), node.opcode(), first, second, leafIndex,
                    node.factor(), row, column);
            }
            return new FastProgram(
                loops[0], loops[1], nodes, outputRow, outputColumn,
                region.scalarProgram().rootNodeId());
        }

        private void execute(Matrix output, Matrix[] leaves) {
            double[][] data = new double[leaves.length][];
            int[] inputColumns = new int[leaves.length];
            for (int index = 0; index < leaves.length; index++) {
                data[index] = leaves[index].getRawData();
                inputColumns[index] = leaves[index].getColumnCount();
            }
            double[] outputData = output.getRawData();
            double[] values = new double[nodes.length];
            int outputColumns = output.getColumnCount();
            for (long first = firstLoop.lowerBound; first < firstLoop.upperBound; first++) {
                for (long second = secondLoop.lowerBound;
                     second < secondLoop.upperBound;
                     second++) {
                    for (FastNode node : nodes) {
                        int nodeId = node.id;
                        switch (node.opcode) {
                            case LOAD -> {
                                int row = checkedIndex(node.row.value(first, second));
                                int column = checkedIndex(node.column.value(first, second));
                                long offset = (long) row * inputColumns[node.leafIndex] + column;
                                values[nodeId] = data[node.leafIndex][checkedIndex(offset)];
                            }
                            case SCALE -> values[nodeId] = node.factor * values[node.firstInput];
                            case ADD -> values[nodeId] = values[node.firstInput]
                                + values[node.secondInput];
                            case TRANSPOSE -> values[nodeId] = values[node.firstInput];
                        }
                    }
                    int row = checkedIndex(outputRow.value(first, second));
                    int column = checkedIndex(outputColumn.value(first, second));
                    long offset = (long) row * outputColumns + column;
                    outputData[checkedIndex(offset)] = values[rootNodeId];
                }
            }
        }
    }

    private static final class FastLoop {
        private final long lowerBound;
        private final long upperBound;

        private FastLoop(long lowerBound, long upperBound) {
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
        }
    }

    private static final class FastNode {
        private final int id;
        private final ScalarFusionProgram.Opcode opcode;
        private final int firstInput;
        private final int secondInput;
        private final int leafIndex;
        private final double factor;
        private final FastIndex row;
        private final FastIndex column;

        private FastNode(int id,
                         ScalarFusionProgram.Opcode opcode,
                         int firstInput,
                         int secondInput,
                         int leafIndex,
                         double factor,
                         FastIndex row,
                         FastIndex column) {
            this.id = id;
            this.opcode = opcode;
            this.firstInput = firstInput;
            this.secondInput = secondInput;
            this.leafIndex = leafIndex;
            this.factor = factor;
            this.row = row;
            this.column = column;
        }
    }

    private static final class FastIndex {
        private final int loopPosition;

        private FastIndex(int loopPosition) {
            this.loopPosition = loopPosition;
        }

        private static FastIndex compile(AffineExpr expression,
                                         Map<AffineVariable, Integer> semanticLoops) {
            if (expression.constant() != 0L || expression.coefficients().size() != 1
                || expression.coefficients().firstEntry().getValue() != 1L) {
                return null;
            }
            Integer loopPosition = semanticLoops.get(expression.coefficients().firstKey());
            return loopPosition == null ? null : new FastIndex(loopPosition);
        }

        private long value(long first, long second) {
            return loopPosition == 0 ? first : second;
        }
    }

    private static boolean inBounds(Matrix matrix, int row, int column) {
        return row >= 0 && row < matrix.getRowCount()
            && column >= 0 && column < matrix.getColumnCount();
    }

    private static int checkedIndex(long value) {
        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
            throw new IllegalStateException(
                "Affine index is outside Java matrix index range: " + value);
        }
        return (int) value;
    }
}
