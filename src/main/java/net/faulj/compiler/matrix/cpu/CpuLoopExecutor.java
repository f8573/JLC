package net.faulj.compiler.matrix.cpu;

import java.util.HashMap;
import java.util.Map;

import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.matrix.Matrix;

/**
 * Bounded realization of the rectangular loop bands emitted by M3.
 *
 * <p>This helper evaluates only affine index expressions attached to the
 * supported explicit loop operations. It is not an interpreter for arbitrary
 * affine programs or expression trees.</p>
 */
final class CpuLoopExecutor {
    @FunctionalInterface
    interface PointBody {
        void accept(Map<AffineVariable, Long> bindings);
    }

    private CpuLoopExecutor() {
    }

    static void forEachPoint(ScheduleBand band, PointBody body) {
        if (band == null || body == null) {
            throw new IllegalArgumentException("CPU loop band and body must not be null");
        }
        visit(band, 0, new HashMap<>(), body);
    }

    static long evaluate(AffineExpr expression, Map<AffineVariable, Long> bindings) {
        long result = expression.constant();
        for (Map.Entry<AffineVariable, Long> term : expression.coefficients().entrySet()) {
            Long value = bindings.get(term.getKey());
            if (value == null) {
                throw new IllegalStateException(
                    "Schedule does not bind affine variable " + term.getKey());
            }
            result = Math.addExact(result, Math.multiplyExact(term.getValue(), value));
        }
        return result;
    }

    static int[] indices(AffineAccess access, Map<AffineVariable, Long> bindings) {
        int[] result = new int[access.indices().size()];
        for (int index = 0; index < result.length; index++) {
            long value = evaluate(access.index(index), bindings);
            if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
                throw new IllegalStateException(
                    "Affine index is outside Java matrix index range: " + value);
            }
            result[index] = (int) value;
        }
        return result;
    }

    static boolean inBounds(AffineAccess access, Matrix matrix, int[] indices) {
        if (indices.length != 2) {
            throw new IllegalStateException(
                "M4 currently supports two-dimensional matrix accesses only");
        }
        return indices[0] >= 0 && indices[0] < matrix.getRowCount()
            && indices[1] >= 0 && indices[1] < matrix.getColumnCount();
    }

    private static void visit(ScheduleBand band,
                              int loopIndex,
                              Map<AffineVariable, Long> bindings,
                              PointBody body) {
        if (loopIndex == band.loops().size()) {
            body.accept(bindings);
            return;
        }

        ScheduleLoop loop = band.loop(loopIndex);
        for (long induction = loop.lowerBound(); induction < loop.upperBound(); induction += loop.step()) {
            bindings.put(loop.inductionVariable(), induction);
            if (loop.valueExpression() == null) {
                bindings.remove(loop.semanticVariable());
            } else {
                long semanticValue = evaluate(loop.valueExpression(), bindings);
                bindings.put(loop.semanticVariable(), semanticValue);
            }
            visit(band, loopIndex + 1, bindings, body);
            if (induction > Long.MAX_VALUE - loop.step()) {
                break;
            }
        }
        bindings.remove(loop.inductionVariable());
        if (loop.valueExpression() != null) {
            bindings.remove(loop.semanticVariable());
        }
    }
}
