package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.affine.AffineVariable;
import net.faulj.compiler.matrix.affine.AliasAnalysis;
import net.faulj.compiler.matrix.affine.AliasRelation;
import net.faulj.compiler.matrix.affine.BufferOwnership;
import net.faulj.compiler.matrix.affine.IterationDomain;
import net.faulj.compiler.matrix.affine.LogicalBuffer;

/** Deterministic verifier for the immutable R3 Kernel IR. */
public final class KernelVerifier {
    private KernelVerifier() {
    }

    public static KernelVerificationResult verify(KernelProgram program) {
        long start = System.nanoTime();
        List<String> diagnostics = new ArrayList<>();
        if (program == null) {
            diagnostics.add("program is null");
            return result(diagnostics, start);
        }
        if (program.id() == null || program.id().isBlank()) {
            diagnostics.add("program has no stable ID");
        }
        if (program.functions().isEmpty()) {
            diagnostics.add("program contains no kernel functions");
        }
        Set<Integer> functionIds = new HashSet<>();
        for (int index = 0; index < program.functions().size(); index++) {
            KernelFunction function = program.functions().get(index);
            if (function == null) {
                diagnostics.add("function[" + index + "] is null");
                continue;
            }
            String prefix = "function " + displayFunction(function) + ": ";
            if (!functionIds.add(function.id())) {
                diagnostics.add(prefix + "duplicate function ID " + function.id());
            }
            verifyFunction(function, prefix, diagnostics);
        }
        return result(diagnostics, start);
    }

    public static KernelVerificationResult verify(KernelFunction function) {
        return verify(function == null ? null : KernelProgram.single(
            function.name() == null ? "unnamed" : "program-" + function.name(), function));
    }

    public static void requireValid(KernelProgram program) {
        KernelVerificationResult result = verify(program);
        if (!result.valid()) {
            throw new KernelVerificationException(result);
        }
    }

    public static void requireValid(KernelFunction function) {
        KernelVerificationResult result = verify(function);
        if (!result.valid()) {
            throw new KernelVerificationException(result);
        }
    }

    private static void verifyFunction(KernelFunction function,
                                       String prefix,
                                       List<String> diagnostics) {
        if (function.id() < 0) {
            diagnostics.add(prefix + "function ID must be non-negative: " + function.id());
        }
        if (function.name() == null || function.name().isBlank()) {
            diagnostics.add(prefix + "missing stable name");
        }

        List<KernelBuffer> inputBuffers = function.inputBuffers();
        List<KernelBuffer> outputBuffers = function.outputBuffers();
        List<KernelBuffer> buffers = function.buffers();
        IdentityHashMap<KernelBuffer, Boolean> bufferSet = new IdentityHashMap<>();
        Map<Integer, KernelBuffer> buffersById = new LinkedHashMap<>();
        for (KernelBuffer buffer : buffers) {
            if (buffer == null) {
                diagnostics.add(prefix + "null buffer descriptor");
                continue;
            }
            if (bufferSet.put(buffer, Boolean.TRUE) != null) {
                diagnostics.add(prefix + "buffer appears more than once: %" + buffer.id());
            }
            if (buffer.id() < 0) {
                diagnostics.add(prefix + "buffer ID must be non-negative: " + buffer.id());
            }
            KernelBuffer previous = buffersById.put(buffer.id(), buffer);
            if (previous != null && previous != buffer) {
                diagnostics.add(prefix + "duplicate buffer ID: " + buffer.id());
            }
            if (buffer.name() == null || buffer.name().isBlank()) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no stable name");
            }
            if (buffer.logicalBuffer() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no logical provenance");
            } else if (buffer.shape() != null
                && !buffer.shape().equals(buffer.logicalBuffer().shape())) {
                diagnostics.add(prefix + "buffer %" + buffer.id()
                    + " shape does not match its logical buffer");
            }
            if (buffer.role() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no role");
            }
            if (buffer.storageType() == null || buffer.computeType() == null
                || buffer.accumulatorType() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has incomplete type facts");
            } else if (buffer.storageType() != KernelValueType.FP64
                || buffer.computeType() != KernelValueType.FP64
                || buffer.accumulatorType() != KernelValueType.FP64) {
                diagnostics.add(prefix + "buffer %" + buffer.id()
                    + " uses unsupported non-FP64 type " + buffer.storageType());
            }
            if (buffer.shape() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no shape");
            }
            if (buffer.layout() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no layout fact");
            }
            if (buffer.memorySpace() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no memory-space fact");
            }
            if (buffer.ownership() == null) {
                diagnostics.add(prefix + "buffer %" + buffer.id() + " has no ownership fact");
            }
        }

        for (KernelBuffer input : inputBuffers) {
            if (input == null) {
                continue;
            }
            if (!bufferSet.containsKey(input)) {
                diagnostics.add(prefix + "input is absent from combined buffer list: %" + input.id());
            }
            if (input.role() != KernelBufferRole.INPUT) {
                diagnostics.add(prefix + "input buffer %" + input.id() + " is not INPUT");
            }
            if (input.logicalBuffer() != null
                && input.logicalBuffer().isSymbolic()
                && input.ownership() != BufferOwnership.NONE) {
                diagnostics.add(prefix + "symbolic input %" + input.id()
                    + " must retain NONE ownership");
            }
        }
        for (KernelBuffer output : outputBuffers) {
            if (output == null) {
                continue;
            }
            if (!bufferSet.containsKey(output)) {
                diagnostics.add(prefix + "output is absent from combined buffer list: %" + output.id());
            }
            if (output.role() != KernelBufferRole.OUTPUT) {
                diagnostics.add(prefix + "output buffer %" + output.id() + " is not OUTPUT");
            }
            if (output.ownership() != BufferOwnership.OWNED) {
                diagnostics.add(prefix + "output buffer %" + output.id()
                    + " is not execution-owned");
            }
        }
        if (outputBuffers.size() != 1) {
            diagnostics.add(prefix + "R3 v1 requires exactly one output buffer");
        }
        for (int first = 0; first < inputBuffers.size(); first++) {
            for (int second = 0; second < outputBuffers.size(); second++) {
                if (inputBuffers.get(first) == outputBuffers.get(second)) {
                    diagnostics.add(prefix + "a buffer cannot be both input and output: %"
                        + inputBuffers.get(first).id());
                }
            }
        }

        Map<AffineVariable, Long> extents = verifyLoops(function, prefix, diagnostics);
        verifyProvenance(function, prefix, diagnostics);
        verifyAliases(function, prefix, diagnostics, bufferSet);
        verifyOperations(function, prefix, diagnostics, bufferSet, extents);
    }

    private static Map<AffineVariable, Long> verifyLoops(KernelFunction function,
                                                          String prefix,
                                                          List<String> diagnostics) {
        Map<AffineVariable, Long> extents = new LinkedHashMap<>();
        IterationDomain domain = function.iterationDomain();
        if (domain == null) {
            diagnostics.add(prefix + "iteration domain is missing");
        }
        if (function.loops().size() != 2) {
            diagnostics.add(prefix + "R3 v1 supports exactly two canonical loops, got "
                + function.loops().size());
        }
        if (domain != null && domain.ranges().size() != function.loops().size()) {
            diagnostics.add(prefix + "loop/domain rank mismatch: loops="
                + function.loops().size() + ", domain=" + domain.ranges().size());
        }
        Map<AffineVariable, IterationDomain.Range> ranges = new LinkedHashMap<>();
        if (domain != null) {
            for (IterationDomain.Range range : domain.ranges()) {
                if (range == null || range.variable() == null) {
                    diagnostics.add(prefix + "iteration domain contains an invalid range");
                    continue;
                }
                ranges.put(range.variable(), range);
                if (range.lowerInclusive() < 0L) {
                    diagnostics.add(prefix + "loop lower bound must be non-negative for "
                        + range.variable());
                }
                try {
                    long extent = Math.subtractExact(
                        range.upperExclusive(), range.lowerInclusive());
                    if (extent < 0L) {
                        diagnostics.add(prefix + "invalid domain bounds for " + range.variable());
                    } else {
                        extents.put(range.variable(), extent);
                    }
                } catch (ArithmeticException overflow) {
                    diagnostics.add(prefix + "domain extent overflows for " + range.variable());
                }
            }
        }

        Set<AffineVariable> induction = new HashSet<>();
        Set<AffineVariable> semantic = new HashSet<>();
        Set<AffineVariable> priorInduction = new HashSet<>();
        for (KernelLoop loop : function.loops()) {
            if (loop == null) {
                diagnostics.add(prefix + "null loop descriptor");
                continue;
            }
            if (!induction.add(loop.inductionVariable())) {
                diagnostics.add(prefix + "duplicate induction variable "
                    + loop.inductionVariable());
            }
            if (!semantic.add(loop.semanticVariable())) {
                diagnostics.add(prefix + "duplicate semantic loop variable "
                    + loop.semanticVariable());
            }
            if (loop.upperBound() < loop.lowerBound()) {
                diagnostics.add(prefix + "invalid loop bounds for " + loop.inductionVariable());
            }
            if (loop.step() <= 0L) {
                diagnostics.add(prefix + "loop step must be positive for "
                    + loop.inductionVariable());
            }
            if (loop.lowerBound() < 0L) {
                diagnostics.add(prefix + "loop lower bound must be non-negative for "
                    + loop.inductionVariable());
            }
            if (!loop.isCanonical()) {
                diagnostics.add(prefix + "non-canonical/tiled loop is unsupported for "
                    + loop.inductionVariable());
            }
            IterationDomain.Range range = ranges.get(loop.semanticVariable());
            if (range == null) {
                diagnostics.add(prefix + "loop variable is absent from the iteration domain: "
                    + loop.semanticVariable());
            } else if (range.lowerInclusive() != loop.lowerBound()
                || range.upperExclusive() != loop.upperBound()) {
                diagnostics.add(prefix + "loop bounds do not match domain for "
                    + loop.semanticVariable());
            }
            if (loop.valueExpression() != null) {
                Set<AffineVariable> visible = new HashSet<>(priorInduction);
                visible.add(loop.inductionVariable());
                for (AffineVariable variable : loop.valueExpression().coefficients().keySet()) {
                    if (!visible.contains(variable)) {
                        diagnostics.add(prefix + "undefined loop variable " + variable
                            + " in binding for " + loop.semanticVariable());
                    }
                }
            }
            priorInduction.add(loop.inductionVariable());
        }
        if (domain != null && !semantic.equals(new HashSet<>(domain.variables()))) {
            diagnostics.add(prefix + "semantic loop variables do not cover the iteration domain");
        }
        return extents;
    }

    private static void verifyProvenance(KernelFunction function,
                                         String prefix,
                                         List<String> diagnostics) {
        KernelProvenance provenance = function.provenance();
        if (provenance == null) {
            diagnostics.add(prefix + "R2 provenance is missing");
            return;
        }
        if (provenance.fusedRegionId() < 0) {
            diagnostics.add(prefix + "source fused-region ID is invalid");
        }
        verifyUniqueNonNegative(provenance.sourceStatementIds(),
            prefix + "source statement IDs", diagnostics, true);
        verifyUniqueNonNegative(provenance.sourceScalarNodeIds(),
            prefix + "source scalar node IDs", diagnostics, true);
        verifyUniqueNonNegative(provenance.eliminatedBufferIds(),
            prefix + "eliminated buffer IDs", diagnostics, true);
        if (provenance.sourceStatementIds() == null
            || provenance.sourceStatementIds().isEmpty()) {
            diagnostics.add(prefix + "source statement provenance is empty");
        }
        if (provenance.sourceScalarNodeIds() == null
            || provenance.sourceScalarNodeIds().isEmpty()) {
            diagnostics.add(prefix + "source scalar provenance is empty");
        }
        // An R2 FusedRegionPlan normally has at least one eliminated
        // materialization, but a standalone kernel may legitimately have no
        // eliminated buffer. Preserve the list without making that case
        // malformed.
    }

    private static void verifyAliases(KernelFunction function,
                                      String prefix,
                                      List<String> diagnostics,
                                      IdentityHashMap<KernelBuffer, Boolean> bufferSet) {
        Map<String, KernelAliasFact> seen = new HashMap<>();
        for (KernelAliasFact fact : function.aliasFacts()) {
            if (fact == null || fact.first() == null || fact.second() == null
                || fact.relation() == null) {
                diagnostics.add(prefix + "invalid alias fact");
                continue;
            }
            if (!bufferSet.containsKey(fact.first()) || !bufferSet.containsKey(fact.second())) {
                diagnostics.add(prefix + "alias fact references an unknown buffer");
                continue;
            }
            if (fact.first() == fact.second()) {
                if (fact.relation() != AliasRelation.MUST_ALIAS) {
                    diagnostics.add(prefix + "self-alias fact must be MUST_ALIAS for %"
                        + fact.first().id());
                }
                continue;
            }
            String key = pairKey(fact.first(), fact.second());
            if (seen.put(key, fact) != null) {
                diagnostics.add(prefix + "duplicate alias fact for " + key);
            }
            LogicalBuffer first = fact.first().logicalBuffer();
            LogicalBuffer second = fact.second().logicalBuffer();
            if (first != null && second != null) {
                AliasRelation actual = AliasAnalysis.between(first, second);
                if (isStrongerThan(fact.relation(), actual)) {
                    diagnostics.add(prefix + "illegal alias strengthening for " + key
                        + ": claimed=" + fact.relation() + ", proven=" + actual);
                }
            }
        }
        List<KernelBuffer> buffers = function.buffers();
        for (int first = 0; first < buffers.size(); first++) {
            for (int second = first + 1; second < buffers.size(); second++) {
                String key = pairKey(buffers.get(first), buffers.get(second));
                if (!seen.containsKey(key)) {
                    diagnostics.add(prefix + "missing alias fact for " + key);
                }
            }
        }
    }

    private static void verifyOperations(KernelFunction function,
                                         String prefix,
                                         List<String> diagnostics,
                                         IdentityHashMap<KernelBuffer, Boolean> bufferSet,
                                         Map<AffineVariable, Long> extents) {
        if (function.body() == null) {
            diagnostics.add(prefix + "kernel body is missing");
            return;
        }
        Map<Integer, KernelValue> values = new LinkedHashMap<>();
        for (KernelValue value : function.values()) {
            if (value == null) {
                diagnostics.add(prefix + "null SSA value declaration");
                continue;
            }
            if (value.id() < 0) {
                diagnostics.add(prefix + "SSA value ID must be non-negative: " + value.id());
            }
            if (values.put(value.id(), value) != null) {
                diagnostics.add(prefix + "duplicate SSA value ID: " + value.id());
            }
            if (value.name() == null || value.name().isBlank()) {
                diagnostics.add(prefix + "SSA value " + value.id() + " has no stable name");
            }
            if (value.type() == null) {
                diagnostics.add(prefix + "SSA value " + value.id() + " has no type");
            } else if (value.type() != KernelValueType.FP64) {
                diagnostics.add(prefix + "SSA value " + value.id()
                    + " uses unsupported type " + value.type());
            }
        }

        Set<Integer> defined = new HashSet<>();
        Set<Integer> sourceScalarIds = function.provenance() == null
            || function.provenance().sourceScalarNodeIds() == null
            ? Set.of() : new HashSet<>(function.provenance().sourceScalarNodeIds());
        Set<Integer> sourceStatementIds = function.provenance() == null
            || function.provenance().sourceStatementIds() == null
            ? Set.of() : new HashSet<>(function.provenance().sourceStatementIds());
        int storeCount = 0;
        KernelAccess outputStoreAccess = null;
        for (int index = 0; index < function.body().operations().size(); index++) {
            KernelOp operation = function.body().operations().get(index);
            String opPrefix = prefix + "op[" + index + "]: ";
            if (operation == null) {
                diagnostics.add(opPrefix + "operation is null");
                continue;
            }
            KernelOpcode opcode = operation.opcode();
            if (opcode == null) {
                diagnostics.add(opPrefix + "unsupported opcode <null>");
                continue;
            }
            if (operation.sourceScalarValueId() < 0
                || !sourceScalarIds.contains(operation.sourceScalarValueId())) {
                diagnostics.add(opPrefix + "source R2 scalar provenance is missing for value "
                    + operation.sourceScalarValueId());
            }
            if (operation.sourceStatementId() < -1
                || operation.sourceStatementId() >= 0
                    && !sourceStatementIds.contains(operation.sourceStatementId())) {
                diagnostics.add(opPrefix + "source statement provenance is invalid: S"
                    + operation.sourceStatementId());
            }

            List<Integer> operands = operation.operands();
            int expectedOperands = switch (opcode) {
                case LOAD, CONSTANT -> 0;
                case ADD, MUL -> 2;
                case STORE -> 1;
            };
            if (operands == null) {
                diagnostics.add(opPrefix + "operand list is missing");
            } else if (operands.size() != expectedOperands) {
                diagnostics.add(opPrefix + "expected " + expectedOperands
                    + " operands, got " + operands.size());
            }
            if (opcode == KernelOpcode.STORE) {
                if (operation.resultValueId() != -1) {
                    diagnostics.add(opPrefix + "STORE must not define an SSA value");
                }
                storeCount++;
            } else {
                if (operation.resultValueId() < 0) {
                    diagnostics.add(opPrefix + "value-producing operation has no result ID");
                } else if (!defined.add(operation.resultValueId())) {
                    diagnostics.add(opPrefix + "duplicate SSA definition: "
                        + operation.resultValueId());
                }
                if (!values.containsKey(operation.resultValueId())) {
                    diagnostics.add(opPrefix + "definition has no declared SSA value: "
                        + operation.resultValueId());
                }
            }

            if (operands != null) {
                for (Integer operand : operands) {
                    if (operand == null) {
                        diagnostics.add(opPrefix + "null SSA operand");
                    } else if (!defined.contains(operand)) {
                        diagnostics.add(opPrefix + "undefined or non-dominating SSA operand: "
                            + operand);
                    }
                }
            }

            switch (opcode) {
                case LOAD -> {
                    if (operation.access() == null) {
                        diagnostics.add(opPrefix + "LOAD has no memory access");
                    } else {
                        verifyAccess(operation.access(), opPrefix, diagnostics, bufferSet, extents);
                        if (operation.access().buffer().role() != KernelBufferRole.INPUT) {
                            diagnostics.add(opPrefix + "LOAD must read an INPUT buffer");
                        }
                        verifyResultType(operation, values, operation.access().buffer().computeType(),
                            opPrefix, diagnostics);
                    }
                    if (operation.immediate() != null) {
                        diagnostics.add(opPrefix + "LOAD must not carry an immediate");
                    }
                }
                case CONSTANT -> {
                    if (operation.immediate() == null) {
                        diagnostics.add(opPrefix + "CONSTANT has no scalar value");
                    }
                    if (operation.access() != null) {
                        diagnostics.add(opPrefix + "CONSTANT must not carry a memory access");
                    }
                    verifyResultType(operation, values, KernelValueType.FP64,
                        opPrefix, diagnostics);
                }
                case ADD, MUL -> {
                    if (operation.access() != null) {
                        diagnostics.add(opPrefix + opcode + " must not carry a memory access");
                    }
                    if (operation.immediate() != null) {
                        diagnostics.add(opPrefix + opcode + " must not carry an immediate");
                    }
                    if (operands != null && operands.size() == 2) {
                        KernelValue first = values.get(operands.get(0));
                        KernelValue second = values.get(operands.get(1));
                        KernelValue result = values.get(operation.resultValueId());
                        if (first != null && second != null && result != null
                            && (first.type() != second.type() || first.type() != result.type())) {
                            diagnostics.add(opPrefix + "scalar type mismatch for " + opcode);
                        }
                    }
                    verifyResultType(operation, values, KernelValueType.FP64,
                        opPrefix, diagnostics);
                }
                case STORE -> {
                    if (operation.access() == null) {
                        diagnostics.add(opPrefix + "STORE has no memory access");
                    } else {
                        verifyAccess(operation.access(), opPrefix, diagnostics, bufferSet, extents);
                        if (operation.access().buffer().role() != KernelBufferRole.OUTPUT) {
                            diagnostics.add(opPrefix + "STORE target is not a writable OUTPUT buffer");
                        }
                        if (outputStoreAccess == null) {
                            outputStoreAccess = operation.access();
                        }
                    }
                    if (operation.immediate() != null) {
                        diagnostics.add(opPrefix + "STORE must not carry an immediate");
                    }
                    if (operands != null && operands.size() == 1) {
                        KernelValue value = values.get(operands.get(0));
                        KernelBuffer target = operation.access() == null
                            ? null : operation.access().buffer();
                        if (value != null && target != null
                            && target.computeType() != value.type()) {
                            diagnostics.add(opPrefix + "STORE scalar type does not match output type");
                        }
                    }
                }
            }
        }
        for (Integer valueId : values.keySet()) {
            if (!defined.contains(valueId)) {
                diagnostics.add(prefix + "declared SSA value is never defined: " + valueId);
            }
        }
        if (storeCount != function.outputBuffers().size()) {
            diagnostics.add(prefix + "output coverage requires exactly one STORE per output; stores="
                + storeCount + ", outputs=" + function.outputBuffers().size());
        }
        if (outputStoreAccess != null && function.outputBuffers().size() == 1) {
            KernelBuffer output = function.outputBuffers().get(0);
            if (outputStoreAccess.buffer() != output) {
                diagnostics.add(prefix + "STORE does not target the declared kernel output");
            }
            Set<AffineVariable> storeVariables = new HashSet<>();
            for (AffineExpr expression : outputStoreAccess.indices()) {
                if (isUnitVariable(expression)) {
                    storeVariables.add(expression.coefficients().firstKey());
                }
            }
            Set<AffineVariable> domainVariables = function.iterationDomain() == null
                ? Set.of() : new HashSet<>(function.iterationDomain().variables());
            if (!storeVariables.equals(domainVariables)) {
                diagnostics.add(prefix + "STORE access is not a bijective output coverage map");
            }
        }
    }

    private static void verifyAccess(KernelAccess access,
                                    String prefix,
                                    List<String> diagnostics,
                                    IdentityHashMap<KernelBuffer, Boolean> bufferSet,
                                    Map<AffineVariable, Long> extents) {
        KernelBuffer buffer = access.buffer();
        if (buffer == null || !bufferSet.containsKey(buffer)) {
            diagnostics.add(prefix + "access references an unknown buffer");
            return;
        }
        if (buffer.shape() == null) {
            diagnostics.add(prefix + "access buffer has no shape");
            return;
        }
        if (access.rank() != 2) {
            diagnostics.add(prefix + "access rank " + access.rank()
                + " does not match matrix rank 2");
            return;
        }
        Set<AffineVariable> used = new HashSet<>();
        for (int dimension = 0; dimension < access.indices().size(); dimension++) {
            AffineExpr expression = access.index(dimension);
            if (!isUnitVariable(expression)) {
                diagnostics.add(prefix + "unsupported affine access expression at dimension "
                    + dimension + ": " + expression);
                continue;
            }
            AffineVariable variable = expression.coefficients().firstKey();
            if (!extents.containsKey(variable)) {
                diagnostics.add(prefix + "undefined loop variable in access: " + variable);
            } else {
                long expected = dimension == 0 ? buffer.shape().rows() : buffer.shape().columns();
                if (extents.get(variable) != expected) {
                    diagnostics.add(prefix + "access extent mismatch for %" + buffer.id()
                        + " dimension " + dimension + ": loop=" + extents.get(variable)
                        + ", shape=" + expected);
                }
            }
            if (!used.add(variable)) {
                diagnostics.add(prefix + "access map is not injective for %" + buffer.id());
            }
        }
    }

    private static void verifyResultType(KernelOp operation,
                                         Map<Integer, KernelValue> values,
                                         KernelValueType expected,
                                         String prefix,
                                         List<String> diagnostics) {
        KernelValue result = values.get(operation.resultValueId());
        if (result != null && result.type() != expected) {
            diagnostics.add(prefix + "result type " + result.type()
                + " does not match required type " + expected);
        }
    }

    private static void verifyUniqueNonNegative(List<Integer> values,
                                                String label,
                                                List<String> diagnostics,
                                                boolean allowNullList) {
        if (values == null) {
            if (!allowNullList) {
                diagnostics.add(label + " is null");
            }
            return;
        }
        Set<Integer> seen = new HashSet<>();
        for (Integer value : values) {
            if (value == null || value < 0) {
                diagnostics.add(label + " contains an invalid ID: " + value);
            } else if (!seen.add(value)) {
                diagnostics.add(label + " contains a duplicate ID: " + value);
            }
        }
    }

    private static boolean isUnitVariable(AffineExpr expression) {
        return expression != null && expression.constant() == 0L
            && expression.coefficients().size() == 1
            && expression.coefficients().firstEntry().getValue() == 1L;
    }

    private static boolean isStrongerThan(AliasRelation claimed, AliasRelation proven) {
        if (claimed == proven) {
            return false;
        }
        if (proven == AliasRelation.MUST_ALIAS) {
            // MAY_ALIAS is a conservative weakening of a known identity.
            return claimed == AliasRelation.NO_ALIAS;
        }
        if (proven == AliasRelation.MAY_ALIAS) {
            return claimed != AliasRelation.MAY_ALIAS;
        }
        return claimed == AliasRelation.MUST_ALIAS;
    }

    private static String pairKey(KernelBuffer first, KernelBuffer second) {
        int left = Math.min(first.id(), second.id());
        int right = Math.max(first.id(), second.id());
        return "%" + left + "<->%" + right;
    }

    private static String displayFunction(KernelFunction function) {
        return function.name() == null ? "<unnamed>" : function.name();
    }

    private static KernelVerificationResult result(List<String> diagnostics, long start) {
        return new KernelVerificationResult(
            diagnostics.isEmpty(), diagnostics, Math.max(0L, System.nanoTime() - start));
    }
}
