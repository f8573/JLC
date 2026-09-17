package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.List;

/** Immutable structural and compiler-time metrics for R3 kernels. */
public final class KernelIrMetrics {
    private final int kernelCount;
    private final int eligibleRegionCount;
    private final int rejectedRegionCount;
    private final int bufferCount;
    private final int loopCount;
    private final int scalarValueCount;
    private final int operationCount;
    private final int loadCount;
    private final int storeCount;
    private final int arithmeticOpCount;
    private final int sharedValueCount;
    private final long loweringTimeNanos;
    private final long verificationTimeNanos;

    private KernelIrMetrics(int kernelCount,
                            int eligibleRegionCount,
                            int rejectedRegionCount,
                            int bufferCount,
                            int loopCount,
                            int scalarValueCount,
                            int operationCount,
                            int loadCount,
                            int storeCount,
                            int arithmeticOpCount,
                            int sharedValueCount,
                            long loweringTimeNanos,
                            long verificationTimeNanos) {
        this.kernelCount = kernelCount;
        this.eligibleRegionCount = eligibleRegionCount;
        this.rejectedRegionCount = rejectedRegionCount;
        this.bufferCount = bufferCount;
        this.loopCount = loopCount;
        this.scalarValueCount = scalarValueCount;
        this.operationCount = operationCount;
        this.loadCount = loadCount;
        this.storeCount = storeCount;
        this.arithmeticOpCount = arithmeticOpCount;
        this.sharedValueCount = sharedValueCount;
        this.loweringTimeNanos = loweringTimeNanos;
        this.verificationTimeNanos = verificationTimeNanos;
    }

    public static KernelIrMetrics empty() {
        return new KernelIrMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0L, 0L);
    }

    public static KernelIrMetrics from(List<KernelLoweringResult> lowerings) {
        if (lowerings == null || lowerings.stream().anyMatch(result -> result == null)) {
            throw new IllegalArgumentException("Kernel lowerings must not be null");
        }
        int kernels = 0;
        int eligible = 0;
        int rejected = 0;
        int buffers = 0;
        int loops = 0;
        int values = 0;
        int operations = 0;
        int loads = 0;
        int stores = 0;
        int arithmetic = 0;
        int shared = 0;
        long loweringNanos = 0L;
        long verificationNanos = 0L;
        for (KernelLoweringResult lowering : lowerings) {
            if (lowering.program() != null) {
                kernels += lowering.program().kernelCount();
                for (KernelFunction function : lowering.program().functions()) {
                    buffers += function.buffers().size();
                    loops += function.loops().size();
                    values += function.values().size();
                    if (function.body() != null) {
                        operations += function.body().operations().size();
                        for (KernelOp operation : function.body().operations()) {
                            if (operation == null || operation.opcode() == null) {
                                continue;
                            }
                            if (operation.opcode() == KernelOpcode.LOAD) {
                                loads++;
                            } else if (operation.opcode() == KernelOpcode.STORE) {
                                stores++;
                            } else if (operation.opcode() == KernelOpcode.ADD
                                || operation.opcode() == KernelOpcode.MUL) {
                                arithmetic++;
                            }
                        }
                    }
                    if (function.body() != null) {
                        shared += sharedValues(function);
                    }
                }
            }
            if (lowering.eligibility() == KernelEligibility.ELIGIBLE
                && lowering.verified()) {
                eligible++;
            } else {
                rejected++;
            }
            loweringNanos = saturatingAdd(loweringNanos, lowering.loweringTimeNanos());
            verificationNanos = saturatingAdd(
                verificationNanos, lowering.verificationTimeNanos());
        }
        return new KernelIrMetrics(
            kernels, eligible, rejected, buffers, loops, values, operations, loads, stores,
            arithmetic, shared, loweringNanos, verificationNanos);
    }

    public int kernelCount() {
        return kernelCount;
    }

    public int eligibleRegionCount() {
        return eligibleRegionCount;
    }

    public int rejectedRegionCount() {
        return rejectedRegionCount;
    }

    public int bufferCount() {
        return bufferCount;
    }

    public int loopCount() {
        return loopCount;
    }

    public int scalarValueCount() {
        return scalarValueCount;
    }

    public int valueCount() {
        return scalarValueCount;
    }

    public int operationCount() {
        return operationCount;
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

    public int sharedValueCount() {
        return sharedValueCount;
    }

    public long loweringTimeNanos() {
        return loweringTimeNanos;
    }

    public long verificationTimeNanos() {
        return verificationTimeNanos;
    }

    public long loweringTimeMicros() {
        return loweringTimeNanos / 1_000L;
    }

    public long verificationTimeMicros() {
        return verificationTimeNanos / 1_000L;
    }

    public String dump() {
        return "kernel IR:\n"
            + "  kernels=" + kernelCount + '\n'
            + "  eligible-regions=" + eligibleRegionCount + '\n'
            + "  rejected-regions=" + rejectedRegionCount + '\n'
            + "  buffers=" + bufferCount + '\n'
            + "  loops=" + loopCount + '\n'
            + "  scalar-values=" + scalarValueCount + '\n'
            + "  operations=" + operationCount + '\n'
            + "  loads=" + loadCount + '\n'
            + "  stores=" + storeCount + '\n'
            + "  arithmetic-ops=" + arithmeticOpCount + '\n'
            + "  shared-values=" + sharedValueCount + '\n'
            + "  lowering-time-ns=" + loweringTimeNanos + '\n'
            + "  verification-time-ns=" + verificationTimeNanos + '\n';
    }

    private static int sharedValues(KernelFunction function) {
        int maxValueId = -1;
        for (KernelValue value : function.values()) {
            maxValueId = Math.max(maxValueId, value.id());
        }
        if (maxValueId < 0) {
            return 0;
        }
        int[] uses = new int[maxValueId + 1];
        for (KernelOp operation : function.body().operations()) {
            if (operation.operands() == null) {
                continue;
            }
            for (Integer operand : operation.operands()) {
                if (operand != null && operand >= 0 && operand < uses.length) {
                    uses[operand]++;
                }
            }
        }
        int result = 0;
        for (int count : uses) {
            if (count > 1) {
                result++;
            }
        }
        return result;
    }

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }
}
