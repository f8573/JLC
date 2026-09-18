package net.faulj.compiler.matrix.codegen;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOpcode;

/**
 * Backend-level facts shared by scalar C++ and AVX2 emitters.
 *
 * <p>The plan is deliberately separate from R3. It records AVX2 policy while
 * retaining the original ordered Kernel IR operations as the single semantic
 * source.</p>
 */
public final class PseudokernelPlan {
    private final KernelFunction function;
    private final KernelSignature signature;
    private final SimdEligibility.Result eligibility;
    private final int vectorWidth;
    private final long elements;
    private final int vectorIterations;
    private final int scalarTailElements;
    private final int loadCount;
    private final int storeCount;
    private final int arithmeticOpCount;
    private final int constantCount;
    private final int maxLiveVectorValues;
    private final long estimatedBytesPerElement;
    private final int arithmeticOpsPerElement;
    private final List<String> rejectionReasons;

    PseudokernelPlan(KernelFunction function,
                     KernelSignature signature,
                     SimdEligibility.Result eligibility) {
        this.function = Objects.requireNonNull(function, "Kernel function");
        this.signature = Objects.requireNonNull(signature, "Kernel signature");
        this.eligibility = Objects.requireNonNull(eligibility, "Eligibility result");
        this.vectorWidth = CodegenBackend.AVX2.vectorWidth();
        long rows = function.loops().get(0).upperBound() - function.loops().get(0).lowerBound();
        long columns = function.loops().get(1).upperBound() - function.loops().get(1).lowerBound();
        this.elements = Math.multiplyExact(rows, columns);
        this.vectorIterations = Math.toIntExact(elements / vectorWidth);
        this.scalarTailElements = Math.toIntExact(elements % vectorWidth);
        int loads = 0;
        int stores = 0;
        int arithmetic = 0;
        int constants = 0;
        for (var operation : function.body().operations()) {
            if (operation.opcode() == KernelOpcode.LOAD) {
                loads++;
            } else if (operation.opcode() == KernelOpcode.STORE) {
                stores++;
            } else if (operation.opcode() == KernelOpcode.ADD
                || operation.opcode() == KernelOpcode.MUL) {
                arithmetic++;
            } else if (operation.opcode() == KernelOpcode.CONSTANT) {
                constants++;
            }
        }
        this.loadCount = loads;
        this.storeCount = stores;
        this.arithmeticOpCount = arithmetic;
        this.constantCount = constants;
        this.maxLiveVectorValues = computeMaxLiveValues(function);
        this.estimatedBytesPerElement = ((long) loads + stores) * Double.BYTES;
        this.arithmeticOpsPerElement = arithmetic;
        this.rejectionReasons = eligibility.rejectionReasons();
    }

    public KernelFunction function() {
        return function;
    }

    public KernelFunction sourceFunction() {
        return function;
    }

    public KernelSignature signature() {
        return signature;
    }

    public SimdEligibility.Result eligibility() {
        return eligibility;
    }

    public boolean avx2Eligible() {
        return eligibility.avx2Eligible();
    }

    public boolean scalarCppEligible() {
        return eligibility.scalarCppEligible();
    }

    public int vectorWidth() {
        return vectorWidth;
    }

    public long elements() {
        return elements;
    }

    public int vectorIterations() {
        return vectorIterations;
    }

    public int scalarTailElements() {
        return scalarTailElements;
    }

    public double vectorizedElementFraction() {
        return elements == 0L ? 1.0 : (double) (vectorIterations * vectorWidth) / elements;
    }

    public int loadCount() {
        return loadCount;
    }

    public int storeCount() {
        return storeCount;
    }

    public int arithmeticOpCount() {
        return arithmeticOpCount;
    }

    public int constantCount() {
        return constantCount;
    }

    public int maxLiveVectorValues() {
        return maxLiveVectorValues;
    }

    public long estimatedBytesPerElement() {
        return estimatedBytesPerElement;
    }

    public int arithmeticOpsPerElement() {
        return arithmeticOpsPerElement;
    }

    public List<String> rejectionReasons() {
        return Collections.unmodifiableList(rejectionReasons);
    }

    public String diagnostic() {
        return "pseudokernel:" + '\n'
            + "  kernel=F" + function.id() + '\n'
            + "  signature=" + signature.shortHash() + '\n'
            + "  target=avx2\n"
            + "  vectorWidth=" + vectorWidth + '\n'
            + "  operations=" + function.body().operations().size() + '\n'
            + "  loads=" + loadCount + '\n'
            + "  stores=" + storeCount + '\n'
            + "  maxLiveVectors=" + maxLiveVectorValues + '\n'
            + "  vectorIterations=" + vectorIterations + '\n'
            + "  tail=" + scalarTailElements + '\n'
            + "  vectorizedFraction=" + vectorizedElementFraction() + '\n'
            + "  eligibility=" + eligibility.status() + '\n'
            + (rejectionReasons.isEmpty() ? "" : "  reason=" + String.join("; ", rejectionReasons) + '\n');
    }

    private static int computeMaxLiveValues(KernelFunction function) {
        int maxId = -1;
        for (var value : function.values()) {
            maxId = Math.max(maxId, value.id());
        }
        if (maxId < 0) {
            return 0;
        }
        int[] lastUse = new int[maxId + 1];
        int[] definition = new int[maxId + 1];
        java.util.Arrays.fill(lastUse, -1);
        java.util.Arrays.fill(definition, -1);
        List<net.faulj.compiler.matrix.kernel.KernelOp> operations = function.body().operations();
        for (int index = 0; index < operations.size(); index++) {
            var operation = operations.get(index);
            if (operation.resultValueId() >= 0 && operation.resultValueId() <= maxId) {
                definition[operation.resultValueId()] = index;
            }
            if (operation.operands() == null) {
                continue;
            }
            for (Integer operand : operation.operands()) {
                if (operand != null && operand >= 0 && operand <= maxId) {
                    lastUse[operand] = index;
                }
            }
        }
        int maxLive = 0;
        for (int index = 0; index < operations.size(); index++) {
            int live = 0;
            for (var operation : operations) {
                int id = operation.resultValueId();
                if (id >= 0 && id <= maxId) {
                    if (definition[id] <= index && lastUse[id] >= index) {
                        live++;
                    }
                }
            }
            maxLive = Math.max(maxLive, live);
        }
        return maxLive;
    }
}
