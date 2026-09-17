package net.faulj.compiler.matrix.codegen;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;

/** Emits one deterministic scalar or AVX2 R5 implementation variant. */
final class VariantCppEmitter {
    private VariantCppEmitter() {
    }

    static GeneratedKernelSource emit(PseudokernelPlan plan,
                                      KernelVariantSignature variant,
                                      CppEmissionOptions options) {
        if (plan == null || variant == null || options == null) {
            throw new IllegalArgumentException("Variant emission arguments are required");
        }
        if (!plan.signature().equals(variant.kernelSignature())) {
            throw new IllegalArgumentException("Variant belongs to a different kernel signature");
        }
        if (variant.backend() == CodegenBackend.AVX2 && !plan.avx2Eligible()) {
            throw new IllegalArgumentException(
                "Kernel is not AVX2 contiguous eligible: " + plan.eligibility().reason());
        }
        if (variant.backend() == CodegenBackend.SCALAR_CPP && !plan.scalarCppEligible()) {
            throw new IllegalArgumentException(
                "Kernel is not eligible for generated scalar C++: " + plan.eligibility().reason());
        }

        long started = System.nanoTime();
        KernelFunction function = plan.function();
        StringBuilder source = new StringBuilder();
        appendPreamble(source, plan, variant, options);
        source.append(CppEmitterSupport.functionSignature(function, variant.generatedSymbol()))
            .append(" {\n");
        appendShapeGuard(source, function);
        Set<Integer> constantIds = constantIds(function);
        if (variant.backend() == CodegenBackend.AVX2) {
            appendVectorConstants(source, function);
            appendAvx2Body(source, plan, variant, constantIds);
        } else {
            appendScalarBody(source, plan, variant);
        }
        source.append("}\n");
        if (options.nativeRegistryRegistration()) {
            CppEmitterSupport.appendRegistryRegistration(source, plan, variant);
        }
        return new GeneratedKernelSource(plan, variant, source.toString(), options,
            System.nanoTime() - started);
    }

    private static void appendPreamble(StringBuilder source,
                                       PseudokernelPlan plan,
                                       KernelVariantSignature variant,
                                       CppEmissionOptions options) {
        source.append("// JLC R5 generated source; deterministic; no timing in identity.\n")
            .append("// kernel-signature-sha256=").append(plan.signature().sha256()).append('\n')
            .append("// variant-sha256=").append(variant.sha256()).append('\n')
            .append("// variant=").append(variant.variantId()).append('\n')
            .append("// provenance=").append(CppEmitterSupport.escape(options.sourceProvenance()))
                .append("\n")
            .append("#include <cstddef>\n#include <cmath>\n#include <limits>\n");
        if (variant.backend() == CodegenBackend.AVX2) {
            source.append("#include <immintrin.h>\n");
        }
        source.append("#pragma STDC FP_CONTRACT OFF\n");
        if (options.nativeRegistryRegistration()) {
            source.append("#include \"jlc_generated_kernel_registry.h\"\n");
        }
        source.append('\n');
    }

    private static void appendShapeGuard(StringBuilder source, KernelFunction function) {
        long rows = function.loops().get(0).upperBound() - function.loops().get(0).lowerBound();
        long columns = function.loops().get(1).upperBound() - function.loops().get(1).lowerBound();
        source.append("    if (rows != ").append(rows).append(" || cols != ")
            .append(columns).append(") return;\n")
            .append("    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) return;\n")
            .append("    const std::size_t count = rows * cols;\n");
    }

    private static void appendVectorConstants(StringBuilder source, KernelFunction function) {
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() == KernelOpcode.CONSTANT) {
                source.append("    const __m256d ")
                    .append(CppEmitterSupport.vectorValueName(operation.resultValueId()))
                    .append(" = _mm256_set1_pd(")
                    .append(CppEmitterSupport.constantLiteral(operation.immediate())).append(");\n");
            }
        }
    }

    private static void appendScalarBody(StringBuilder source,
                                         PseudokernelPlan plan,
                                         KernelVariantSignature variant) {
        KernelFunction function = plan.function();
        if (variant.loopForm() == KernelLoopForm.FLAT) {
            source.append("    for (std::size_t p = 0; p < count; ++p) {\n");
            for (KernelOp operation : function.body().operations()) {
                appendScalarOperation(source, function, operation, true, "p", null);
            }
            source.append("    }\n");
            return;
        }
        String outer = function.loops().get(0).inductionVariable().name();
        String inner = function.loops().get(1).inductionVariable().name();
        source.append("    for (std::size_t ").append(outer).append(" = 0; ")
            .append(outer).append(" < rows; ++").append(outer).append(") {\n")
            .append("        for (std::size_t ").append(inner).append(" = 0; ")
            .append(inner).append(" < cols; ++").append(inner).append(") {\n");
        for (KernelOp operation : function.body().operations()) {
            appendScalarOperation(source, function, operation, false, outer, inner);
        }
        source.append("        }\n    }\n");
    }

    private static void appendAvx2Body(StringBuilder source,
                                       PseudokernelPlan plan,
                                       KernelVariantSignature variant,
                                       Set<Integer> constantIds) {
        KernelFunction function = plan.function();
        Map<Integer, KernelOp> fusedMuls = fmaPairs(function, variant);
        Set<Integer> consumedMuls = new HashSet<>(fusedMuls.keySet());
        if (variant.loopForm() == KernelLoopForm.FLAT) {
            source.append("    std::size_t p = 0;\n");
            appendVectorLoop(source, function, variant, constantIds, fusedMuls, consumedMuls,
                "    ", "p", "count", "    ");
            return;
        }
        String outer = function.loops().get(0).inductionVariable().name();
        String inner = function.loops().get(1).inductionVariable().name();
        source.append("    for (std::size_t ").append(outer).append(" = 0; ")
            .append(outer).append(" < rows; ++").append(outer).append(") {\n")
            .append("        std::size_t ").append(inner).append(" = 0;\n");
        String indent = "        ";
        appendVectorLoop(source, function, variant, constantIds, fusedMuls, consumedMuls,
            indent, "p", "cols", indent, inner, outer);
        source.append(indent).append("for (; ").append(inner).append(" < cols; ++")
            .append(inner).append(") {\n");
        for (KernelOp operation : function.body().operations()) {
            appendScalarOperation(source, function, operation, false, outer, inner);
        }
        source.append(indent).append("}\n    }\n");
    }

    private static void appendVectorLoop(StringBuilder source,
                                         KernelFunction function,
                                         KernelVariantSignature variant,
                                         Set<Integer> constantIds,
                                         Map<Integer, KernelOp> fusedMuls,
                                         Set<Integer> consumedMuls,
                                         String indent,
                                         String pointerVariable,
                                         String bound,
                                         String bodyIndent) {
        appendVectorLoop(source, function, variant, constantIds, fusedMuls, consumedMuls,
            indent, pointerVariable, bound, bodyIndent, null, null);
    }

    private static void appendVectorLoop(StringBuilder source,
                                         KernelFunction function,
                                         KernelVariantSignature variant,
                                         Set<Integer> constantIds,
                                         Map<Integer, KernelOp> fusedMuls,
                                         Set<Integer> consumedMuls,
                                         String indent,
                                         String pointerVariable,
                                         String bound,
                                         String bodyIndent,
                                         String innerVariable,
                                         String outerVariable) {
        int width = variant.vectorWidth();
        int unroll = variant.unroll();
        int stride = width * unroll;
        String loopVariable = innerVariable == null ? pointerVariable : innerVariable;
        String pointer = innerVariable == null ? pointerVariable : "p";
        source.append(indent).append("for (; ").append(loopVariable).append(" + ")
            .append(stride).append(" <= ").append(bound).append("; ")
            .append(loopVariable).append(" += ").append(stride).append(") {\n");
        if (innerVariable != null) {
            source.append(bodyIndent).append("    const std::size_t p = ")
                .append(outerVariable).append(" * cols + ").append(innerVariable).append(";\n");
        }
        for (int group = 0; group < unroll; group++) {
            for (KernelOp operation : function.body().operations()) {
                if (operation.opcode() == KernelOpcode.CONSTANT
                    || (operation.opcode() == KernelOpcode.MUL
                        && consumedMuls.contains(operation.resultValueId()))) {
                    continue;
                }
                appendVectorOperation(source, function, operation, group, constantIds,
                    pointer, bodyIndent + "    ", fusedMuls);
            }
        }
        source.append(indent).append("}\n");
        source.append(indent).append("for (; ").append(loopVariable).append(" < ")
            .append(bound).append("; ++").append(loopVariable).append(") {\n");
        if (innerVariable != null) {
            source.append(bodyIndent).append("    const std::size_t p = ")
                .append(outerVariable).append(" * cols + ").append(innerVariable).append(";\n");
        }
        for (KernelOp operation : function.body().operations()) {
            appendScalarOperation(source, function, operation, true, "p", null, bodyIndent + "    ", fusedMuls);
        }
        source.append(indent).append("}\n");
    }

    private static void appendVectorOperation(StringBuilder source,
                                              KernelFunction function,
                                              KernelOp operation,
                                              int group,
                                              Set<Integer> constantIds,
                                              String pointer,
                                              String indent,
                                              Map<Integer, KernelOp> fusedMuls) {
        String result = vectorName(operation.resultValueId(), group, constantIds);
        String suffix = " + " + pointer + " + " + (group * 4);
        switch (operation.opcode()) {
            case LOAD -> source.append(indent).append("const __m256d ").append(result)
                .append(" = _mm256_loadu_pd(")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append(suffix).append(");\n");
            case ADD -> {
                KernelOp mul = fusedMuls.get(operation.resultValueId());
                if (mul != null) {
                    int other = operation.operands().get(0) == mul.resultValueId()
                        ? operation.operands().get(1) : operation.operands().get(0);
                    source.append(indent).append("const __m256d ").append(result)
                        .append(" = _mm256_fmadd_pd(")
                        .append(vectorName(mul.operands().get(0), group, constantIds)).append(", ")
                        .append(vectorName(mul.operands().get(1), group, constantIds)).append(", ")
                        .append(vectorName(other, group, constantIds)).append(");\n");
                } else {
                    appendBinary(source, indent, result, "_mm256_add_pd", operation, group, constantIds);
                }
            }
            case MUL -> appendBinary(source, indent, result, "_mm256_mul_pd", operation,
                group, constantIds);
            case STORE -> source.append(indent).append("_mm256_storeu_pd(")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append(suffix).append(", ").append(vectorName(operation.operands().get(0),
                    group, constantIds)).append(");\n");
            case CONSTANT -> {
                // Hoisted above the loop.
            }
        }
    }

    private static void appendBinary(StringBuilder source,
                                     String indent,
                                     String result,
                                     String intrinsic,
                                     KernelOp operation,
                                     int group,
                                     Set<Integer> constantIds) {
        source.append(indent).append("const __m256d ").append(result).append(" = ")
            .append(intrinsic).append('(')
            .append(vectorName(operation.operands().get(0), group, constantIds)).append(", ")
            .append(vectorName(operation.operands().get(1), group, constantIds)).append(");\n");
    }

    private static void appendScalarOperation(StringBuilder source,
                                              KernelFunction function,
                                              KernelOp operation,
                                              boolean flat,
                                              String outer,
                                              String inner) {
        appendScalarOperation(source, function, operation, flat, outer, inner, "        ", Map.of());
    }

    private static void appendScalarOperation(StringBuilder source,
                                              KernelFunction function,
                                              KernelOp operation,
                                              boolean flat,
                                              String outer,
                                              String inner,
                                              String indent,
                                              Map<Integer, KernelOp> fusedMuls) {
        String result = CppEmitterSupport.scalarValueName(operation.resultValueId());
        String access = operation.access() == null ? null
            : flat
                ? CppEmitterSupport.bufferParameter(function, operation.access().buffer()) + "[p]"
                : CppEmitterSupport.scalarAccess(function, operation.access(), outer, inner,
                    function.loops().get(0).inductionVariable().name(),
                    function.loops().get(1).inductionVariable().name());
        switch (operation.opcode()) {
            case LOAD -> source.append(indent).append("const double ").append(result)
                .append(" = ").append(access).append(";\n");
            case CONSTANT -> source.append(indent).append("const double ").append(result)
                .append(" = ").append(CppEmitterSupport.constantLiteral(operation.immediate()))
                .append(";\n");
            case ADD -> {
                KernelOp mul = fusedMuls.get(operation.resultValueId());
                if (mul != null) {
                    int other = operation.operands().get(0) == mul.resultValueId()
                        ? operation.operands().get(1) : operation.operands().get(0);
                    source.append(indent).append("const double ").append(result)
                        .append(" = std::fma(")
                        .append(CppEmitterSupport.scalarValueName(mul.operands().get(0))).append(", ")
                        .append(CppEmitterSupport.scalarValueName(mul.operands().get(1))).append(", ")
                        .append(CppEmitterSupport.scalarValueName(other)).append(");\n");
                } else {
                    appendScalarBinary(source, indent, result, "+", operation);
                }
            }
            case MUL -> {
                if (!fusedMuls.containsKey(operation.resultValueId())) {
                    appendScalarBinary(source, indent, result, "*", operation);
                }
            }
            case STORE -> source.append(indent)
                .append(flat ? CppEmitterSupport.bufferParameter(function, operation.access().buffer()) + "[p]"
                    : CppEmitterSupport.scalarAccess(function, operation.access(), outer, inner,
                        function.loops().get(0).inductionVariable().name(),
                        function.loops().get(1).inductionVariable().name()))
                .append(" = ").append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(";\n");
        }
    }

    private static void appendScalarBinary(StringBuilder source,
                                           String indent,
                                           String result,
                                           String operator,
                                           KernelOp operation) {
        source.append(indent).append("const double ").append(result).append(" = ")
            .append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
            .append(' ').append(operator).append(' ')
            .append(CppEmitterSupport.scalarValueName(operation.operands().get(1))).append(";\n");
    }

    private static String vectorName(int valueId, int group, Set<Integer> constants) {
        String base = CppEmitterSupport.vectorValueName(valueId);
        return constants.contains(valueId) ? base : base + "_u" + group;
    }

    private static Set<Integer> constantIds(KernelFunction function) {
        Set<Integer> result = new HashSet<>();
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() == KernelOpcode.CONSTANT) {
                result.add(operation.resultValueId());
            }
        }
        return result;
    }

    private static Map<Integer, KernelOp> fmaPairs(KernelFunction function,
                                                   KernelVariantSignature variant) {
        if (variant.fmaContraction() != FmaContraction.EXPLICIT) {
            return Map.of();
        }
        Map<Integer, KernelOp> muls = new HashMap<>();
        Map<Integer, Integer> uses = new HashMap<>();
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() == KernelOpcode.MUL) {
                muls.put(operation.resultValueId(), operation);
            }
            if (operation.operands() != null) {
                for (Integer operand : operation.operands()) {
                    uses.merge(operand, 1, Integer::sum);
                }
            }
        }
        Map<Integer, KernelOp> result = new HashMap<>();
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() != KernelOpcode.ADD || operation.operands() == null) {
                continue;
            }
            for (Integer operand : operation.operands()) {
                KernelOp mul = muls.get(operand);
                if (mul != null && uses.getOrDefault(operand, 0) == 1) {
                    result.put(operation.resultValueId(), mul);
                    break;
                }
            }
        }
        return result;
    }
}
