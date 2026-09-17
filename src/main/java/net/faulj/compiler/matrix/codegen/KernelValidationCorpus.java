package net.faulj.compiler.matrix.codegen;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.matrix.Matrix;

/** Deterministic ordinary and IEEE-edge bindings for a shape-specialized kernel. */
public final class KernelValidationCorpus {
    private KernelValidationCorpus() {
    }

    public static List<KernelBinding> forPlan(PseudokernelPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("Pseudokernel plan must not be null");
        }
        KernelFunction function = plan.function();
        List<KernelBinding> result = new ArrayList<>();
        result.add(binding(function, false));
        result.add(binding(function, true));
        return List.copyOf(result);
    }

    private static KernelBinding binding(KernelFunction function, boolean edgeValues) {
        IdentityHashMap<KernelBuffer, Matrix> matrices = new IdentityHashMap<>();
        for (KernelBuffer buffer : function.inputBuffers()) {
            matrices.put(buffer, matrix(buffer, edgeValues));
        }
        for (KernelBuffer buffer : function.outputBuffers()) {
            matrices.put(buffer, new Matrix(buffer.shape().rows(), buffer.shape().columns()));
        }
        return new KernelBinding(function, matrices);
    }

    private static Matrix matrix(KernelBuffer buffer, boolean edgeValues) {
        int rows = buffer.shape().rows();
        int columns = buffer.shape().columns();
        double[] data = new double[Math.multiplyExact(rows, columns)];
        for (int index = 0; index < data.length; index++) {
            if (edgeValues) {
                data[index] = edgeValue(index);
            } else {
                data[index] = finiteValue(index, buffer.id());
            }
        }
        return Matrix.wrap(data, rows, columns);
    }

    private static double finiteValue(int index, int bufferId) {
        long x = 0x9E3779B97F4A7C15L ^ (long) index * 0xBF58476D1CE4E5B9L
            ^ (long) bufferId * 0x94D049BB133111EBL;
        x = (x ^ (x >>> 30)) * 0xBF58476D1CE4E5B9L;
        x = (x ^ (x >>> 27)) * 0x94D049BB133111EBL;
        x ^= x >>> 31;
        return ((x & 0xFFFFL) - 32768L) / 8192.0 + 0.125;
    }

    private static double edgeValue(int index) {
        return switch (index & 15) {
            case 0 -> Double.NaN;
            case 1 -> Double.POSITIVE_INFINITY;
            case 2 -> Double.NEGATIVE_INFINITY;
            case 3 -> 0.0;
            case 4 -> -0.0;
            case 5 -> Double.MIN_VALUE;
            case 6 -> -Double.MIN_VALUE;
            case 7 -> Double.MAX_VALUE;
            case 8 -> -Double.MAX_VALUE;
            case 9 -> 1.0e-300;
            case 10 -> -1.0e-300;
            default -> finiteValue(index + 17, 11);
        };
    }
}
