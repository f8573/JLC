package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import net.faulj.compiler.matrix.affine.AliasRelation;
import net.faulj.compiler.matrix.affine.MemorySpace;
import net.faulj.compiler.matrix.kernel.KernelAccess;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelLayout;
import net.faulj.compiler.matrix.kernel.KernelLoop;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;
import net.faulj.compiler.matrix.kernel.KernelValueType;
import net.faulj.compiler.matrix.kernel.KernelVerifier;

/** Conservative structural AVX2 eligibility analysis for verified R3 IR. */
public final class SimdEligibility {
    public enum Status {
        AVX2_CONTIGUOUS,
        SCALAR_CPP,
        INELIGIBLE
    }

    public static final class Result {
        private final Status status;
        private final List<String> rejectionReasons;

        private Result(Status status, List<String> rejectionReasons) {
            this.status = status;
            this.rejectionReasons = Collections.unmodifiableList(new ArrayList<>(rejectionReasons));
        }

        public Status status() {
            return status;
        }

        public boolean avx2Eligible() {
            return status == Status.AVX2_CONTIGUOUS;
        }

        public boolean scalarCppEligible() {
            return status == Status.AVX2_CONTIGUOUS || status == Status.SCALAR_CPP;
        }

        public List<String> rejectionReasons() {
            return rejectionReasons;
        }

        public String reason() {
            return rejectionReasons.isEmpty() ? "eligible" : String.join("; ", rejectionReasons);
        }

        @Override
        public String toString() {
            return status + (rejectionReasons.isEmpty() ? "" : " (" + reason() + ")");
        }
    }

    private SimdEligibility() {
    }

    public static Result analyze(KernelFunction function) {
        if (function == null) {
            return ineligible("kernel function is null");
        }
        try {
            KernelVerifier.requireValid(function);
        } catch (RuntimeException failure) {
            return ineligible("verified Kernel IR required: " + message(failure));
        }
        List<String> scalarReasons = new ArrayList<>();
        List<String> avx2Reasons = new ArrayList<>();
        if (function.outputBuffers().size() != 1) {
            return ineligible("R4 requires exactly one output buffer");
        }
        KernelBuffer output = function.outputBuffers().get(0);
        for (KernelBuffer buffer : function.buffers()) {
            if (buffer.storageType() != KernelValueType.FP64
                || buffer.computeType() != KernelValueType.FP64
                || buffer.accumulatorType() != KernelValueType.FP64) {
                return ineligible("only real FP64 buffers are supported");
            }
            if (buffer.layout() != KernelLayout.ROW_MAJOR_DENSE) {
                scalarReasons.add("buffer %" + buffer.id() + " is not row-major dense");
                avx2Reasons.add("buffer %" + buffer.id() + " is not row-major dense");
            }
            boolean runtimeOwnedOutput = buffer.isOutput()
                && buffer.memorySpace() == MemorySpace.UNKNOWN;
            if (buffer.memorySpace() != MemorySpace.HEAP && !runtimeOwnedOutput) {
                scalarReasons.add("buffer %" + buffer.id()
                    + " is not in the initial heap storage subset");
                avx2Reasons.add("buffer %" + buffer.id()
                    + " is not in the initial heap storage subset");
            }
        }
        KernelLoop outer = function.loops().get(0);
        KernelLoop inner = function.loops().get(1);
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() != KernelOpcode.LOAD
                && operation.opcode() != KernelOpcode.STORE) {
                continue;
            }
            KernelAccess access = operation.access();
            if (!isContiguous(access, outer, inner)) {
                String kind = operation.opcode() == KernelOpcode.LOAD ? "load" : "store";
                avx2Reasons.add("non-unit-stride " + kind + " " + access);
            }
        }
        for (KernelBuffer input : function.inputBuffers()) {
            if (function.aliasRelation(output, input) != AliasRelation.NO_ALIAS) {
                avx2Reasons.add("output %" + output.id() + " and input %" + input.id()
                    + " do not have proven NO_ALIAS");
            }
        }
        if (!avx2Reasons.isEmpty()) {
            if (!scalarReasons.isEmpty()) {
                return new Result(Status.INELIGIBLE, merge(scalarReasons, avx2Reasons));
            }
            return scalarCpp(avx2Reasons);
        }
        return new Result(Status.AVX2_CONTIGUOUS, List.of());
    }

    private static boolean isContiguous(KernelAccess access, KernelLoop outer, KernelLoop inner) {
        return access != null && access.rank() == 2
            && isVariable(access.index(0), outer.inductionVariable())
            && isVariable(access.index(1), inner.inductionVariable());
    }

    private static boolean isVariable(net.faulj.compiler.matrix.affine.AffineExpr expression,
                                      net.faulj.compiler.matrix.affine.AffineVariable variable) {
        return expression != null && expression.constant() == 0L
            && expression.coefficients().size() == 1
            && expression.coefficient(variable) == 1L;
    }

    private static Result scalarCpp(List<String> reasons) {
        return new Result(Status.SCALAR_CPP, reasons);
    }

    private static Result ineligible(String reason) {
        return new Result(Status.INELIGIBLE, List.of(reason));
    }

    private static List<String> merge(List<String> first, List<String> second) {
        List<String> result = new ArrayList<>(first);
        result.addAll(second);
        return result;
    }

    private static String message(RuntimeException failure) {
        return failure.getMessage() == null
            ? failure.getClass().getSimpleName() : failure.getMessage();
    }
}
