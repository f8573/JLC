package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;

/** Emits explicit AVX2 intrinsics plus a scalar remainder from the same plan. */
public final class Avx2CppEmitter {
    private Avx2CppEmitter() {
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan) {
        return emit(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan,
                                             CppEmissionOptions options) {
        CppEmitterSupport.requirePlan(plan);
        if (!plan.avx2Eligible()) {
            throw new IllegalArgumentException(
                "Kernel is not AVX2 contiguous eligible: " + plan.eligibility().reason());
        }
        long start = System.nanoTime();
        KernelFunction function = plan.function();
        String symbol = plan.signature().generatedSymbol();
        long expectedRows = function.loops().get(0).upperBound()
            - function.loops().get(0).lowerBound();
        long expectedColumns = function.loops().get(1).upperBound()
            - function.loops().get(1).lowerBound();
        StringBuilder source = new StringBuilder();
        appendPreamble(source, plan, options);
        source.append(CppEmitterSupport.functionSignature(function, symbol)).append(" {\n");
        for (KernelOp operation : function.body().operations()) {
            if (operation.opcode() == KernelOpcode.CONSTANT) {
                source.append("    const __m256d ")
                    .append(CppEmitterSupport.vectorValueName(operation.resultValueId()))
                    .append(" = _mm256_set1_pd(")
                    .append(CppEmitterSupport.constantLiteral(operation.immediate())).append(");\n");
            }
        }
        source.append(
            "    if (rows != ").append(expectedRows).append(" || cols != ")
            .append(expectedColumns).append(") return;\n")
            .append("    if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols) return;\n")
            .append("    const std::size_t count = rows * cols;\n")
            .append("    std::size_t p = 0;\n")
            .append("    for (; p + 4 <= count; p += 4) {\n");
        for (KernelOp operation : function.body().operations()) {
            appendVectorOperation(source, function, operation);
        }
        source.append("    }\n")
            .append("    for (; p < count; ++p) {\n");
        for (KernelOp operation : function.body().operations()) {
            appendScalarTailOperation(source, function, operation);
        }
        source.append("    }\n}\n");
        if (options.nativeRegistryRegistration()) {
            CppEmitterSupport.appendRegistryRegistration(source, plan, CodegenBackend.AVX2);
        }
        return new GeneratedKernelSource(plan, CodegenBackend.AVX2, source.toString(),
            options, System.nanoTime() - start);
    }

    private static void appendPreamble(StringBuilder source,
                                       PseudokernelPlan plan,
                                       CppEmissionOptions options) {
        source.append("// JLC R4 generated source; deterministic; no timestamp.\n")
            .append("// signature-sha256=").append(plan.signature().sha256()).append('\n')
            .append("// backend=avx2\n")
            .append("// provenance=").append(CppEmitterSupport.escape(options.sourceProvenance()))
                .append("\n")
            .append("#include <cstddef>\n#include <immintrin.h>\n#include <limits>\n")
            .append("#pragma STDC FP_CONTRACT OFF\n");
        if (options.nativeRegistryRegistration()) {
            source.append("#include \"jlc_generated_kernel_registry.h\"\n");
        }
        source.append('\n');
    }

    private static void appendVectorOperation(StringBuilder source,
                                              KernelFunction function,
                                              KernelOp operation) {
        String result = CppEmitterSupport.vectorValueName(operation.resultValueId());
        switch (operation.opcode()) {
            case LOAD -> source.append("        const __m256d ").append(result)
                .append(" = _mm256_loadu_pd(")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append(" + p);\n");
            case CONSTANT -> {
                // Hoisted before the loop; the definition is intentionally not repeated.
            }
            case ADD -> source.append("        const __m256d ").append(result)
                .append(" = _mm256_add_pd(")
                .append(CppEmitterSupport.vectorValueName(operation.operands().get(0)))
                .append(", ").append(CppEmitterSupport.vectorValueName(operation.operands().get(1)))
                .append(");\n");
            case MUL -> source.append("        const __m256d ").append(result)
                .append(" = _mm256_mul_pd(")
                .append(CppEmitterSupport.vectorValueName(operation.operands().get(0)))
                .append(", ").append(CppEmitterSupport.vectorValueName(operation.operands().get(1)))
                .append(");\n");
            case STORE -> source.append("        _mm256_storeu_pd(")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append(" + p, ")
                .append(CppEmitterSupport.vectorValueName(operation.operands().get(0)))
                .append(");\n");
        }
    }

    private static void appendScalarTailOperation(StringBuilder source,
                                                  KernelFunction function,
                                                  KernelOp operation) {
        String result = CppEmitterSupport.scalarValueName(operation.resultValueId());
        switch (operation.opcode()) {
            case LOAD -> source.append("        const double ").append(result).append(" = ")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append("[p];\n");
            case CONSTANT -> source.append("        const double ").append(result).append(" = ")
                .append(CppEmitterSupport.constantLiteral(operation.immediate())).append(";\n");
            case ADD -> source.append("        const double ").append(result).append(" = ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(" + ").append(CppEmitterSupport.scalarValueName(operation.operands().get(1)))
                .append(";\n");
            case MUL -> source.append("        const double ").append(result).append(" = ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(" * ").append(CppEmitterSupport.scalarValueName(operation.operands().get(1)))
                .append(";\n");
            case STORE -> source.append("        ")
                .append(CppEmitterSupport.bufferParameter(function, operation.access().buffer()))
                .append("[p] = ").append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(";\n");
        }
    }
}
