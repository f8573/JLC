package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;

/** Emits strict scalar C++ from the common R4 pseudokernel plan. */
public final class ScalarCppEmitter {
    private ScalarCppEmitter() {
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan) {
        return emit(plan, CppEmissionOptions.standalone());
    }

    public static GeneratedKernelSource emit(PseudokernelPlan plan,
                                             CppEmissionOptions options) {
        CppEmitterSupport.requirePlan(plan);
        long start = System.nanoTime();
        KernelFunction function = plan.function();
        String symbol = plan.signature().generatedSymbol();
        String outerVariable = function.loops().get(0).inductionVariable().name();
        String innerVariable = function.loops().get(1).inductionVariable().name();
        StringBuilder source = new StringBuilder();
        appendPreamble(source, plan, CodegenBackend.SCALAR_CPP, options);
        long expectedRows = function.loops().get(0).upperBound()
            - function.loops().get(0).lowerBound();
        long expectedColumns = function.loops().get(1).upperBound()
            - function.loops().get(1).lowerBound();
        source.append(CppEmitterSupport.functionSignature(function, symbol)).append(" {\n")
            .append("    if (rows != ").append(expectedRows).append(" || cols != ")
            .append(expectedColumns).append(") return;\n")
            .append("    for (std::size_t i = 0; i < rows; ++i) {\n")
            .append("        for (std::size_t j = 0; j < cols; ++j) {\n");
        for (KernelOp operation : function.body().operations()) {
            appendOperation(source, function, operation, "i", "j", outerVariable, innerVariable);
        }
        source.append("        }\n    }\n}\n");
        if (options.nativeRegistryRegistration()) {
            CppEmitterSupport.appendRegistryRegistration(source, plan, CodegenBackend.SCALAR_CPP);
        }
        return new GeneratedKernelSource(plan, CodegenBackend.SCALAR_CPP, source.toString(),
            options, System.nanoTime() - start);
    }

    private static void appendPreamble(StringBuilder source,
                                       PseudokernelPlan plan,
                                       CodegenBackend backend,
                                       CppEmissionOptions options) {
        source.append("// JLC R4 generated source; deterministic; no timestamp.\n")
            .append("// signature-sha256=").append(plan.signature().sha256()).append('\n')
            .append("// backend=").append(backend.propertyValue()).append('\n')
            .append("// provenance=").append(CppEmitterSupport.escape(options.sourceProvenance()))
                .append("\n")
            .append("#include <cstddef>\n#include <limits>\n")
            .append("#pragma STDC FP_CONTRACT OFF\n");
        if (options.nativeRegistryRegistration()) {
            source.append("#include \"jlc_generated_kernel_registry.h\"\n");
        }
        source.append('\n');
    }

    private static void appendOperation(StringBuilder source,
                                        KernelFunction function,
                                        KernelOp operation,
                                        String outer,
                                        String inner,
                                        String outerVariable,
                                        String innerVariable) {
        String result = CppEmitterSupport.scalarValueName(operation.resultValueId());
        switch (operation.opcode()) {
            case LOAD -> source.append("            const double ").append(result).append(" = ")
                .append(CppEmitterSupport.scalarAccess(function, operation.access(), outer, inner,
                    outerVariable, innerVariable))
                .append(";\n");
            case CONSTANT -> source.append("            const double ").append(result).append(" = ")
                .append(CppEmitterSupport.constantLiteral(operation.immediate())).append(";\n");
            case ADD -> source.append("            const double ").append(result).append(" = ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(" + ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(1)))
                .append(";\n");
            case MUL -> source.append("            const double ").append(result).append(" = ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(" * ")
                .append(CppEmitterSupport.scalarValueName(operation.operands().get(1)))
                .append(";\n");
            case STORE -> source.append("            ")
                .append(CppEmitterSupport.scalarAccess(function, operation.access(), outer, inner,
                    outerVariable, innerVariable))
                .append(" = ").append(CppEmitterSupport.scalarValueName(operation.operands().get(0)))
                .append(";\n");
        }
    }
}
