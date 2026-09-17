package net.faulj.compiler.matrix.cpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import net.faulj.compiler.matrix.MatrixShape;
import net.faulj.compiler.matrix.affine.AccessKind;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.StatementKind;

/**
 * Deterministic, estimated fusion accounting for one final CPU plan.
 *
 * <p>Traffic values are logical payload estimates, not hardware DRAM
 * measurements.  Fusion elimination and R1 physical reuse are intentionally
 * represented by different fields.</p>
 */
public final class FusionMetrics {
    private final FusionStrategy strategy;
    private final int candidateRegionCount;
    private final int acceptedRegionCount;
    private final int rejectedRegionCount;
    private final int fusedStatementCount;
    private final int fusedRegionCount;
    private final int fusionElidedTemporaryCount;
    private final long fusionElidedBytes;
    private final int fullMatrixPassesBefore;
    private final int fullMatrixPassesAfter;
    private final long estimatedLogicalBytesReadBefore;
    private final long estimatedLogicalBytesReadAfter;
    private final long estimatedLogicalBytesWrittenBefore;
    private final long estimatedLogicalBytesWrittenAfter;
    private final int largestRegionStatementCount;
    private final int largestRegionScalarNodeCount;
    private final int internalScalarNodeCount;
    private final int internalScalarUseCount;
    private final int sharedScalarProducerCount;
    private final List<String> decisions;

    private FusionMetrics(FusionStrategy strategy,
                          int candidateRegionCount,
                          int acceptedRegionCount,
                          int rejectedRegionCount,
                          int fusedStatementCount,
                          int fusedRegionCount,
                          int fusionElidedTemporaryCount,
                          long fusionElidedBytes,
                          int fullMatrixPassesBefore,
                          int fullMatrixPassesAfter,
                          long estimatedLogicalBytesReadBefore,
                          long estimatedLogicalBytesReadAfter,
                          long estimatedLogicalBytesWrittenBefore,
                          long estimatedLogicalBytesWrittenAfter,
                          int largestRegionStatementCount,
                          int largestRegionScalarNodeCount,
                          int internalScalarNodeCount,
                          int internalScalarUseCount,
                          int sharedScalarProducerCount,
                          List<String> decisions) {
        this.strategy = Objects.requireNonNull(strategy, "Fusion strategy must not be null");
        this.candidateRegionCount = requireNonNegative(candidateRegionCount, "Candidate count");
        this.acceptedRegionCount = requireNonNegative(acceptedRegionCount, "Accepted count");
        this.rejectedRegionCount = requireNonNegative(rejectedRegionCount, "Rejected count");
        this.fusedStatementCount = requireNonNegative(fusedStatementCount, "Fused statement count");
        this.fusedRegionCount = requireNonNegative(fusedRegionCount, "Fused region count");
        this.fusionElidedTemporaryCount = requireNonNegative(
            fusionElidedTemporaryCount, "Fusion-elided temporary count");
        this.fusionElidedBytes = requireNonNegative(fusionElidedBytes, "Fusion-elided bytes");
        this.fullMatrixPassesBefore = requireNonNegative(
            fullMatrixPassesBefore, "Pass count before");
        this.fullMatrixPassesAfter = requireNonNegative(fullMatrixPassesAfter, "Pass count after");
        this.estimatedLogicalBytesReadBefore = requireNonNegative(
            estimatedLogicalBytesReadBefore, "Read estimate before");
        this.estimatedLogicalBytesReadAfter = requireNonNegative(
            estimatedLogicalBytesReadAfter, "Read estimate after");
        this.estimatedLogicalBytesWrittenBefore = requireNonNegative(
            estimatedLogicalBytesWrittenBefore, "Write estimate before");
        this.estimatedLogicalBytesWrittenAfter = requireNonNegative(
            estimatedLogicalBytesWrittenAfter, "Write estimate after");
        this.largestRegionStatementCount = requireNonNegative(
            largestRegionStatementCount, "Largest region statement count");
        this.largestRegionScalarNodeCount = requireNonNegative(
            largestRegionScalarNodeCount, "Largest region scalar node count");
        this.internalScalarNodeCount = requireNonNegative(
            internalScalarNodeCount, "Internal scalar node count");
        this.internalScalarUseCount = requireNonNegative(
            internalScalarUseCount, "Internal scalar use count");
        this.sharedScalarProducerCount = requireNonNegative(
            sharedScalarProducerCount, "Shared scalar producer count");
        if (decisions == null || decisions.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Fusion decisions must not be null");
        }
        this.decisions = Collections.unmodifiableList(new ArrayList<>(decisions));
    }

    /** Build accounting after the final CPU steps have been selected. */
    static FusionMetrics create(FusionStrategy strategy,
                                AffineProgram program,
                                List<CpuStep> steps,
                                List<FusedRegionPlan> acceptedRegions,
                                List<LogicalBuffer> elidedBuffers,
                                int candidateRegionCount,
                                int rejectedRegionCount,
                                List<String> decisions) {
        if (program == null || steps == null || acceptedRegions == null || elidedBuffers == null) {
            throw new IllegalArgumentException("Fusion metric inputs must not be null");
        }
        List<FusedRegionPlan> regions = List.copyOf(acceptedRegions);
        List<LogicalBuffer> elided = List.copyOf(elidedBuffers);
        int fusedStatementCount = regions.stream()
            .mapToInt(region -> region.statements().size()).sum();
        int fusedRegionCount = regions.size();
        int legacyRegionCount = (int) steps.stream()
            .filter(step -> step instanceof CpuFusedElementwiseStep)
            .count();
        if (legacyRegionCount > 0) {
            fusedRegionCount += legacyRegionCount;
            fusedStatementCount += legacyRegionCount * 2;
        }
        int acceptedCount = fusedRegionCount;
        int candidates = Math.max(candidateRegionCount, acceptedCount);
        long elidedBytes = 0L;
        for (LogicalBuffer buffer : elided) {
            elidedBytes = add(elidedBytes, payloadBytes(buffer.shape()));
        }

        Traffic before = trafficBefore(program);
        Traffic after = trafficAfter(steps);
        int largestStatements = regions.stream()
            .mapToInt(region -> region.statements().size()).max().orElse(legacyRegionCount > 0 ? 2 : 0);
        int largestNodes = regions.stream()
            .mapToInt(region -> region.scalarProgram().nodeCount()).max().orElse(0);
        int internalNodes = regions.stream()
            .mapToInt(region -> region.scalarProgram().internalNodeCount()).sum();
        int internalUses = regions.stream()
            .mapToInt(region -> region.scalarProgram().scalarUseCount()).sum();
        int sharedProducers = regions.stream()
            .mapToInt(region -> region.scalarProgram().sharedProducerCount()).sum();

        return new FusionMetrics(
            strategy,
            candidates,
            acceptedCount,
            Math.max(0, rejectedRegionCount),
            fusedStatementCount,
            fusedRegionCount,
            elided.size(),
            elidedBytes,
            before.passes,
            steps.size(),
            before.reads,
            after.reads,
            before.writes,
            after.writes,
            largestStatements,
            largestNodes,
            internalNodes,
            internalUses,
            sharedProducers,
            decisions == null ? List.of() : decisions);
    }

    static FusionMetrics empty(FusionStrategy strategy,
                               AffineProgram program,
                               List<CpuStep> steps) {
        return create(strategy, program, steps, List.of(), List.of(), 0, 0, List.of());
    }

    public FusionStrategy strategy() {
        return strategy;
    }

    public int candidateRegionCount() {
        return candidateRegionCount;
    }

    public int getCandidateRegionCount() {
        return candidateRegionCount();
    }

    public int acceptedRegionCount() {
        return acceptedRegionCount;
    }

    public int getAcceptedRegionCount() {
        return acceptedRegionCount();
    }

    public int rejectedRegionCount() {
        return rejectedRegionCount;
    }

    public int getRejectedRegionCount() {
        return rejectedRegionCount();
    }

    public int fusedStatementCount() {
        return fusedStatementCount;
    }

    public int fusedRegionCount() {
        return fusedRegionCount;
    }

    public int fusionElidedTemporaryCount() {
        return fusionElidedTemporaryCount;
    }

    public int elidedTemporaryCount() {
        return fusionElidedTemporaryCount();
    }

    public long fusionElidedBytes() {
        return fusionElidedBytes;
    }

    public int fullMatrixPassesBefore() {
        return fullMatrixPassesBefore;
    }

    public int fullMatrixPassesAfter() {
        return fullMatrixPassesAfter;
    }

    public long estimatedLogicalBytesReadBefore() {
        return estimatedLogicalBytesReadBefore;
    }

    public long estimatedLogicalBytesReadAfter() {
        return estimatedLogicalBytesReadAfter;
    }

    public long estimatedLogicalBytesWrittenBefore() {
        return estimatedLogicalBytesWrittenBefore;
    }

    public long estimatedLogicalBytesWrittenAfter() {
        return estimatedLogicalBytesWrittenAfter;
    }

    public int largestRegionStatementCount() {
        return largestRegionStatementCount;
    }

    public int largestRegionScalarNodeCount() {
        return largestRegionScalarNodeCount;
    }

    public int internalScalarNodeCount() {
        return internalScalarNodeCount;
    }

    public int internalScalarUseCount() {
        return internalScalarUseCount;
    }

    public int sharedScalarProducerCount() {
        return sharedScalarProducerCount;
    }

    public List<String> decisions() {
        return decisions;
    }

    public String dump() {
        StringBuilder result = new StringBuilder("fusion metrics:\n");
        result.append("  strategy=").append(strategy.propertyValue()).append('\n');
        result.append("  candidateRegionCount=").append(candidateRegionCount).append('\n');
        result.append("  acceptedRegionCount=").append(acceptedRegionCount).append('\n');
        result.append("  rejectedRegionCount=").append(rejectedRegionCount).append('\n');
        result.append("  fusedStatementCount=").append(fusedStatementCount).append('\n');
        result.append("  fusedRegionCount=").append(fusedRegionCount).append('\n');
        result.append("  fusionElidedTemporaryCount=").append(fusionElidedTemporaryCount).append('\n');
        result.append("  fusionElidedBytes=").append(fusionElidedBytes).append('\n');
        result.append("  fullMatrixPassesBefore=").append(fullMatrixPassesBefore).append('\n');
        result.append("  fullMatrixPassesAfter=").append(fullMatrixPassesAfter).append('\n');
        result.append("  estimatedLogicalBytesReadBefore=")
            .append(estimatedLogicalBytesReadBefore).append('\n');
        result.append("  estimatedLogicalBytesReadAfter=")
            .append(estimatedLogicalBytesReadAfter).append('\n');
        result.append("  estimatedLogicalBytesWrittenBefore=")
            .append(estimatedLogicalBytesWrittenBefore).append('\n');
        result.append("  estimatedLogicalBytesWrittenAfter=")
            .append(estimatedLogicalBytesWrittenAfter).append('\n');
        result.append("  largestRegionStatementCount=").append(largestRegionStatementCount).append('\n');
        result.append("  largestRegionScalarNodeCount=").append(largestRegionScalarNodeCount).append('\n');
        result.append("  internalScalarNodeCount=").append(internalScalarNodeCount).append('\n');
        result.append("  internalScalarUseCount=").append(internalScalarUseCount).append('\n');
        result.append("  sharedScalarProducerCount=").append(sharedScalarProducerCount).append('\n');
        if (!decisions.isEmpty()) {
            result.append("  decisions:\n");
            for (String decision : decisions) {
                result.append("    ").append(decision).append('\n');
            }
        }
        return result.toString();
    }

    private static Traffic trafficBefore(AffineProgram program) {
        int passes = 0;
        long reads = 0L;
        long writes = 0L;
        for (AffineStatement statement : program.statements()) {
            if (statement.kind() != StatementKind.MATMUL_UPDATE) {
                passes++;
            }
            for (var access : statement.accesses()) {
                if (access.kind() == AccessKind.REDUCTION) {
                    continue;
                }
                if (access.reads()) {
                    reads = add(reads, payloadBytes(access.buffer().shape()));
                }
                if (access.writes()) {
                    writes = add(writes, payloadBytes(access.buffer().shape()));
                }
            }
        }
        return new Traffic(passes, reads, writes);
    }

    private static Traffic trafficAfter(List<CpuStep> steps) {
        long reads = 0L;
        long writes = 0L;
        for (CpuStep step : steps) {
            if (step instanceof CpuFusedRegionStep fused) {
                for (ScalarFusionProgram.ScalarNode node : fused.regionPlan().scalarProgram().nodes()) {
                    if (node.isLoad()) {
                        reads = add(reads, payloadBytes(node.buffer().shape()));
                    }
                }
                writes = add(writes, payloadBytes(step.outputBuffer().shape()));
            } else if (step instanceof CpuGemmStep gemm) {
                reads = add(reads, payloadBytes(gemm.lhs().shape()));
                reads = add(reads, payloadBytes(gemm.rhs().shape()));
                writes = add(writes, payloadBytes(gemm.outputBuffer().shape()));
            } else {
                for (LogicalBuffer input : step.inputBuffers()) {
                    reads = add(reads, payloadBytes(input.shape()));
                }
                if (step.outputBuffer() != null) {
                    writes = add(writes, payloadBytes(step.outputBuffer().shape()));
                }
            }
        }
        return new Traffic(steps.size(), reads, writes);
    }

    private static long payloadBytes(MatrixShape shape) {
        try {
            return Math.multiplyExact(
                Math.multiplyExact((long) shape.rows(), shape.columns()), Double.BYTES);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    private static long add(long first, long second) {
        if (first == Long.MAX_VALUE || second == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        try {
            return Math.addExact(first, second);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    private static int requireNonNegative(int value, String role) {
        if (value < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
        return value;
    }

    private static long requireNonNegative(long value, String role) {
        if (value < 0) {
            throw new IllegalArgumentException(role + " must be non-negative");
        }
        return value;
    }

    private record Traffic(int passes, long reads, long writes) {
    }
}
