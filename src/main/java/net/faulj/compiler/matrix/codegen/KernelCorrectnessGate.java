package net.faulj.compiler.matrix.codegen;

import java.util.IdentityHashMap;
import java.util.List;

import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelReferenceExecutor;
import net.faulj.matrix.Matrix;

/** Runs candidate implementations against the verified R3 reference oracle. */
public final class KernelCorrectnessGate {
    private KernelCorrectnessGate() {
    }

    public static KernelCorrectnessResult validate(PseudokernelPlan plan,
                                                   GeneratedKernelInvoker candidate,
                                                   List<KernelBinding> corpus) {
        return validate(plan, candidate, corpus, OptimizationSemantics.STRICT);
    }

    public static KernelCorrectnessResult validate(PseudokernelPlan plan,
                                                   GeneratedKernelInvoker candidate,
                                                   List<KernelBinding> corpus,
                                                   OptimizationSemantics semantics) {
        if (plan == null || candidate == null || corpus == null || semantics == null) {
            throw new IllegalArgumentException("Correctness gate arguments must not be null");
        }
        KernelFunction function = plan.function();
        int checked = 0;
        for (KernelBinding fixture : corpus) {
            if (fixture == null || fixture.function() != function) {
                return new KernelCorrectnessResult(false, checked,
                    "validation fixture belongs to a different kernel function");
            }
            try {
                KernelBinding referenceBinding = copyBinding(function, fixture);
                Matrix expected = KernelReferenceExecutor.executeVerified(function, referenceBinding);
                KernelBinding candidateBinding = copyBinding(function, fixture);
                if (!candidate.invoke(candidateBinding)) {
                    return new KernelCorrectnessResult(false, checked,
                        "candidate returned false during validation");
                }
                Matrix actual = candidateBinding.matrix(function.outputBuffers().get(0));
                String mismatch = compare(expected, actual, semantics);
                if (mismatch != null) {
                    return new KernelCorrectnessResult(false, checked,
                        "case " + checked + ": " + mismatch);
                }
                checked++;
            } catch (RuntimeException | Error failure) {
                return new KernelCorrectnessResult(false, checked,
                    "case " + checked + " threw " + failure.getClass().getSimpleName()
                        + ": " + message(failure));
            }
        }
        return new KernelCorrectnessResult(true, checked, "PASS");
    }

    private static KernelBinding copyBinding(KernelFunction function, KernelBinding original) {
        IdentityHashMap<KernelBuffer, Matrix> matrices = new IdentityHashMap<>();
        for (KernelBuffer buffer : function.buffers()) {
            Matrix matrix = original.matrix(buffer);
            if (matrix == null) {
                throw new IllegalArgumentException("unbound validation buffer %" + buffer.id());
            }
            matrices.put(buffer, Matrix.wrap(matrix.getRawData().clone(),
                matrix.getRowCount(), matrix.getColumnCount()));
        }
        return new KernelBinding(function, matrices);
    }

    private static String compare(Matrix expected,
                                  Matrix actual,
                                  OptimizationSemantics semantics) {
        if (actual == null || expected.getRawData().length != actual.getRawData().length) {
            return "output shape or binding is missing";
        }
        double[] left = expected.getRawData();
        double[] right = actual.getRawData();
        for (int index = 0; index < left.length; index++) {
            double expectedValue = left[index];
            double actualValue = right[index];
            if (Double.isNaN(expectedValue) && Double.isNaN(actualValue)) {
                continue;
            }
            if (semantics == OptimizationSemantics.STRICT) {
                if (Double.doubleToRawLongBits(expectedValue)
                    != Double.doubleToRawLongBits(actualValue)) {
                    return "bit mismatch at element " + index + ": expected="
                        + expectedValue + " actual=" + actualValue;
                }
            } else {
                double scale = Math.max(1.0, Math.max(Math.abs(expectedValue), Math.abs(actualValue)));
                if (Double.isFinite(expectedValue) && Double.isFinite(actualValue)
                    && Math.abs(expectedValue - actualValue) > 1.0e-12 * scale) {
                    return "numeric mismatch at element " + index + ": expected="
                        + expectedValue + " actual=" + actualValue;
                }
                if (Double.isInfinite(expectedValue) && expectedValue != actualValue) {
                    return "infinity mismatch at element " + index;
                }
            }
        }
        return null;
    }

    private static String message(Throwable failure) {
        return failure.getMessage() == null ? "" : failure.getMessage();
    }
}
