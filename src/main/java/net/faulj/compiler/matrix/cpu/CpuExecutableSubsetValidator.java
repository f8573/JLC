package net.faulj.compiler.matrix.cpu;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.faulj.compiler.matrix.affine.AccessKind;
import net.faulj.compiler.matrix.affine.AffineAccess;
import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineProgram;
import net.faulj.compiler.matrix.affine.AffineStatement;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.affine.StatementKind;
import net.faulj.compiler.matrix.schedule.ScheduleBand;
import net.faulj.compiler.matrix.schedule.ScheduleLoop;
import net.faulj.compiler.matrix.schedule.ScheduleRegion;
import net.faulj.compiler.matrix.schedule.ScheduleSequence;
import net.faulj.compiler.matrix.schedule.ScheduleStatement;

/** Rejection boundary for the deliberately small M4 executable IR subset. */
final class CpuExecutableSubsetValidator {
    private CpuExecutableSubsetValidator() {
    }

    static void validate(net.faulj.compiler.matrix.schedule.SchedulePlan schedule) {
        AffineProgram program = schedule.program();
        Set<LogicalBuffer> programBuffers = Collections.newSetFromMap(new IdentityHashMap<>());
        programBuffers.addAll(program.buffers());
        for (AffineStatement statement : program.statements()) {
            validateStatement(statement, programBuffers);
        }
        if (program.resultBuffer().isTemporary()
            && program.statements().stream().flatMap(statement -> statement.accesses().stream())
                .noneMatch(access -> access.buffer() == program.resultBuffer() && access.writes())) {
            throw new IllegalArgumentException(
                "Unsupported CPU executable IR: result buffer has no complete producer");
        }
        for (ScheduleRegion region : schedule.regions()) {
            validateBody(region);
            validateBand(region.band(), region.statements().get(0));
        }
    }

    private static void validateStatement(AffineStatement statement,
                                          Set<LogicalBuffer> programBuffers) {
        if (statement.kind() == StatementKind.SYNTHETIC) {
            reject(statement, "synthetic statements are not executable");
        }
        for (AffineAccess access : statement.accesses()) {
            if (!programBuffers.contains(access.buffer())) {
                reject(statement, "an access refers to a foreign logical buffer");
            }
            if (access.indices().size() != 2) {
                reject(statement, "M4 supports only two-dimensional accesses");
            }
            if (access.writes() && access.buffer().isExternalInput()) {
                reject(statement, "writes to borrowed external buffers are forbidden");
            }
        }
        switch (statement.kind()) {
            case ADD -> validateElementwise(statement, List.of(
                AccessKind.WRITE, AccessKind.READ, AccessKind.READ));
            case SCALE -> validateElementwise(statement, List.of(
                AccessKind.WRITE, AccessKind.READ));
            case TRANSPOSE -> validateTranspose(statement);
            case MATMUL_INIT -> validateMatMulInit(statement);
            case MATMUL_UPDATE -> validateMatMulUpdate(statement);
            default -> reject(statement, "unsupported statement kind " + statement.kind());
        }
    }

    private static void validateElementwise(AffineStatement statement,
                                            List<AccessKind> kinds) {
        requireDomainRank(statement, 2);
        requireKinds(statement, kinds);
        AffineVariable i = statement.domain().variables().get(0);
        AffineVariable j = statement.domain().variables().get(1);
        for (AffineAccess access : statement.accesses()) {
            requireIndices(statement, access, variable(i), variable(j));
            requireDomainMatchesBuffer(statement, access.buffer(), false);
        }
    }

    private static void validateTranspose(AffineStatement statement) {
        requireDomainRank(statement, 2);
        requireKinds(statement, List.of(AccessKind.WRITE, AccessKind.READ));
        AffineVariable i = statement.domain().variables().get(0);
        AffineVariable j = statement.domain().variables().get(1);
        AffineAccess write = statement.accesses().get(0);
        AffineAccess read = statement.accesses().get(1);
        requireIndices(statement, write, variable(j), variable(i));
        requireIndices(statement, read, variable(i), variable(j));
        requireDomainMatchesBuffer(statement, read.buffer(), false);
        requireDomainMatchesBuffer(statement, write.buffer(), true);
    }

    private static void validateMatMulInit(AffineStatement statement) {
        requireDomainRank(statement, 2);
        requireKinds(statement, List.of(AccessKind.WRITE));
        AffineVariable i = statement.domain().variables().get(0);
        AffineVariable j = statement.domain().variables().get(1);
        requireIndices(statement, statement.accesses().get(0), variable(i), variable(j));
        requireDomainMatchesBuffer(statement, statement.accesses().get(0).buffer(), false);
    }

    private static void validateMatMulUpdate(AffineStatement statement) {
        requireDomainRank(statement, 3);
        requireKinds(statement, List.of(AccessKind.READ, AccessKind.READ, AccessKind.REDUCTION));
        if (!statement.hasReduction()) {
            reject(statement, "MATMUL_UPDATE is missing reduction metadata");
        }
        AffineVariable i = statement.domain().variables().get(0);
        AffineVariable j = statement.domain().variables().get(1);
        AffineVariable k = statement.domain().variables().get(2);
        if (!statement.reduction().variable().equals(k)) {
            reject(statement, "MATMUL_UPDATE reduction variable does not match its domain");
        }
        requireIndices(statement, statement.accesses().get(0), variable(i), variable(k));
        requireIndices(statement, statement.accesses().get(1), variable(k), variable(j));
        requireIndices(statement, statement.accesses().get(2), variable(i), variable(j));
        long rows = extent(statement.domain(), i);
        long columns = extent(statement.domain(), j);
        long reduction = extent(statement.domain(), k);
        requireCanonicalRange(statement, i, statement.accesses().get(2).buffer().shape().rows());
        requireCanonicalRange(statement, j, statement.accesses().get(2).buffer().shape().columns());
        requireCanonicalRange(statement, k, statement.accesses().get(0).buffer().shape().columns());
        if (statement.accesses().get(0).buffer().shape().rows() != rows
            || statement.accesses().get(0).buffer().shape().columns() != reduction
            || statement.accesses().get(1).buffer().shape().rows() != reduction
            || statement.accesses().get(1).buffer().shape().columns() != columns
            || statement.accesses().get(2).buffer().shape().rows() != rows
            || statement.accesses().get(2).buffer().shape().columns() != columns) {
            reject(statement, "MATMUL access shapes do not match its MxKxN domain");
        }
    }

    private static void validateBody(ScheduleRegion region) {
        if (region.isSingleton()) {
            if (!(region.band().body() instanceof ScheduleStatement leaf)
                || leaf.statement() != region.statements().get(0)) {
                throw new IllegalArgumentException(
                    "CPU lowering rejects nested or malformed singleton schedule bodies");
            }
            return;
        }
        if (!(region.band().body() instanceof ScheduleSequence sequence)
            || sequence.children().size() != region.statements().size()) {
            throw new IllegalArgumentException(
                "CPU lowering rejects unsupported nested fused schedule bodies");
        }
        for (int index = 0; index < sequence.children().size(); index++) {
            if (!(sequence.children().get(index) instanceof ScheduleStatement leaf)
                || leaf.statement() != region.statements().get(index)) {
                throw new IllegalArgumentException(
                    "CPU lowering rejects unsupported nested fused schedule bodies");
            }
        }
    }

    private static void validateBand(ScheduleBand band, AffineStatement statement) {
        Set<AffineVariable> domainVariables = new HashSet<>(statement.domain().variables());
        Set<AffineVariable> inductionVariables = new HashSet<>();
        Set<AffineVariable> available = new HashSet<>();
        Map<AffineVariable, Integer> bindings = new HashMap<>();
        Map<AffineVariable, ScheduleLoop> precedingLoops = new HashMap<>();
        for (ScheduleLoop loop : band.loops()) {
            if (!inductionVariables.add(loop.inductionVariable())) {
                reject(statement, "schedule induction variables must be unique");
            }
            available.add(loop.inductionVariable());
            AffineExpr value = loop.valueExpression();
            if (value == null) {
                if (loop.guard() != null) {
                    reject(statement, "an unbound schedule loop cannot carry a guard");
                }
                precedingLoops.put(loop.inductionVariable(), loop);
                continue;
            }
            if (!available.containsAll(value.coefficients().keySet())) {
                reject(statement, "a derived index is used before its binder loop");
            }
            if (!domainVariables.contains(loop.semanticVariable())) {
                reject(statement, "a schedule binding targets a foreign domain variable");
            }
            bindings.merge(loop.semanticVariable(), 1, Integer::sum);
            available.add(loop.semanticVariable());
            validateBinding(loop, statement, precedingLoops);
            precedingLoops.put(loop.inductionVariable(), loop);
        }
        for (AffineVariable variable : domainVariables) {
            if (bindings.getOrDefault(variable, 0) != 1) {
                reject(statement, "schedule does not bind each domain variable exactly once");
            }
        }
        for (ScheduleLoop loop : band.loops()) {
            if (loop.valueExpression() != null) {
                continue;
            }
            long uses = band.loops().stream()
                .filter(candidate -> candidate.valueExpression() != null)
                .filter(candidate -> candidate.valueExpression().coefficient(
                    loop.inductionVariable()) != 0L)
                .count();
            if (uses != 1L) {
                reject(statement, "an unbound loop must be exactly one generated tile binder");
            }
        }
    }

    private static void validateBinding(ScheduleLoop loop,
                                        AffineStatement statement,
                                        Map<AffineVariable, ScheduleLoop> precedingLoops) {
        IterationDomain.Range range = statement.domain().ranges().stream()
            .filter(candidate -> candidate.variable().equals(loop.semanticVariable()))
            .findFirst().orElseThrow();
        AffineExpr canonical = variable(loop.inductionVariable());
        if (loop.valueExpression().equals(canonical)) {
            if (!loop.inductionVariable().equals(loop.semanticVariable())
                || loop.lowerBound() != range.lowerInclusive()
                || loop.upperBound() != range.upperExclusive()
                || loop.step() != 1L || loop.guard() != null) {
                reject(statement, "malformed canonical schedule loop");
            }
            return;
        }
        if (loop.step() != 1L || loop.guard() == null
            || !loop.guard().equals(loop.semanticVariable() + " < " + range.upperExclusive())
            || loop.valueExpression().coefficient(loop.inductionVariable()) != 1L
            || loop.valueExpression().coefficients().size() != 2
            || loop.lowerBound() != 0L) {
            reject(statement, "unsupported schedule guard or binding");
        }
        AffineVariable outerVariable = loop.valueExpression().coefficients().keySet().stream()
            .filter(variable -> !variable.equals(loop.inductionVariable()))
            .findFirst().orElseThrow();
        long tileSize = loop.valueExpression().coefficient(outerVariable);
        ScheduleLoop outer = precedingLoops.get(outerVariable);
        long extent = range.upperExclusive() - range.lowerInclusive();
        long expectedOuter = extent == 0L ? 0L : (extent + tileSize - 1L) / tileSize;
        if (tileSize <= 0L || loop.upperBound() != tileSize
            || loop.valueExpression().constant() != range.lowerInclusive()
            || outer == null || outer.valueExpression() != null
            || !outer.semanticVariable().equals(loop.semanticVariable())
            || outer.lowerBound() != 0L || outer.upperBound() != expectedOuter
            || outer.step() != 1L) {
            reject(statement, "unsupported or malformed tile binding composition");
        }
    }

    private static void requireDomainRank(AffineStatement statement, int rank) {
        if (statement.domain().ranges().size() != rank) {
            reject(statement, "expected a complete rank-" + rank + " rectangular domain");
        }
    }

    private static void requireKinds(AffineStatement statement, List<AccessKind> expected) {
        List<AccessKind> actual = statement.accesses().stream().map(AffineAccess::kind).toList();
        if (!actual.equals(expected)) {
            reject(statement, "access effects do not match the executable " + statement.kind() + " form");
        }
    }

    private static void requireIndices(AffineStatement statement,
                                       AffineAccess access,
                                       AffineExpr first,
                                       AffineExpr second) {
        if (!access.indices().equals(List.of(first, second))) {
            reject(statement, "access indices are outside the executable " + statement.kind() + " form");
        }
    }

    private static void requireDomainMatchesBuffer(AffineStatement statement,
                                                   LogicalBuffer buffer,
                                                   boolean transposed) {
        long first = extent(statement.domain(), statement.domain().variables().get(0));
        long second = extent(statement.domain(), statement.domain().variables().get(1));
        long rows = transposed ? second : first;
        long columns = transposed ? first : second;
        requireCanonicalRange(statement, statement.domain().variables().get(0),
            transposed ? buffer.shape().columns() : buffer.shape().rows());
        requireCanonicalRange(statement, statement.domain().variables().get(1),
            transposed ? buffer.shape().rows() : buffer.shape().columns());
        if (buffer.shape().rows() != rows || buffer.shape().columns() != columns) {
            reject(statement, "domain does not completely cover access buffer shape");
        }
    }

    private static void requireCanonicalRange(AffineStatement statement,
                                              AffineVariable variable,
                                              long expectedExtent) {
        IterationDomain.Range range = statement.domain().ranges().stream()
            .filter(candidate -> candidate.variable().equals(variable))
            .findFirst().orElseThrow();
        if (range.lowerInclusive() != 0L || range.upperExclusive() != expectedExtent) {
            reject(statement, "domain must cover the exact canonical [0,N) logical range");
        }
    }

    private static long extent(IterationDomain domain, AffineVariable variable) {
        return domain.extent(variable);
    }

    private static AffineExpr variable(AffineVariable variable) {
        return AffineExpr.variable(variable);
    }

    private static void reject(AffineStatement statement, String reason) {
        throw new IllegalArgumentException(
            "Unsupported CPU executable IR at S" + statement.id() + ": " + reason);
    }
}
