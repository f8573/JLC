package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.affine.AffineExpr;
import net.faulj.compiler.matrix.kernel.KernelAccess;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.compiler.matrix.kernel.KernelOp;
import net.faulj.compiler.matrix.kernel.KernelOpcode;

/** Package-private deterministic formatting helpers shared by both emitters. */
final class CppEmitterSupport {
    private CppEmitterSupport() {
    }

    static void requirePlan(PseudokernelPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("Pseudokernel plan must not be null");
        }
        if (!plan.scalarCppEligible()) {
            throw new IllegalArgumentException(
                "Kernel is not eligible for generated scalar C++: " + plan.eligibility().reason());
        }
    }

    static String valueName(int id) {
        return "k" + id;
    }

    static String vectorValueName(int id) {
        return "v" + id;
    }

    static String scalarValueName(int id) {
        return valueName(id);
    }

    static String bufferParameter(KernelFunction function, KernelBuffer buffer) {
        int inputIndex = function.inputBuffers().indexOf(buffer);
        if (inputIndex >= 0) {
            return "b" + inputIndex;
        }
        if (function.outputBuffers().get(0) == buffer) {
            return "out";
        }
        throw new IllegalArgumentException("Unknown kernel buffer %" + buffer.id());
    }

    static String constantLiteral(double value) {
        if (Double.isNaN(value)) {
            return "std::numeric_limits<double>::quiet_NaN()";
        }
        if (value == Double.POSITIVE_INFINITY) {
            return "std::numeric_limits<double>::infinity()";
        }
        if (value == Double.NEGATIVE_INFINITY) {
            return "-std::numeric_limits<double>::infinity()";
        }
        String result = Double.toString(value);
        if (!result.contains(".") && !result.contains("e") && !result.contains("E")) {
            result += ".0";
        }
        return result;
    }

    static String scalarAccess(KernelFunction function,
                               KernelAccess access,
                               String outer,
                               String inner) {
        return scalarAccess(function, access, outer, inner,
            function.loops().get(0).inductionVariable().name(),
            function.loops().get(1).inductionVariable().name());
    }

    static String scalarAccess(KernelFunction function,
                               KernelAccess access,
                               String outer,
                               String inner,
                               String outerVariable,
                               String innerVariable) {
        KernelBuffer buffer = access.buffer();
        String row = affineIndex(access.index(0), outer, inner, outerVariable, innerVariable);
        String column = affineIndex(access.index(1), outer, inner, outerVariable, innerVariable);
        return bufferParameter(function, buffer) + "[(" + row + ") * "
            + buffer.shape().columns() + " + (" + column + ")]";
    }

    static String affineIndex(AffineExpr expression, String outer, String inner) {
        return affineIndex(expression, outer, inner, "i", "j");
    }

    static String affineIndex(AffineExpr expression,
                              String outer,
                              String inner,
                              String outerVariable,
                              String innerVariable) {
        if (expression.constant() == 0L && expression.coefficients().size() == 1) {
            var term = expression.coefficients().firstEntry();
            if (term.getValue() == 1L) {
                if (term.getKey().name().equals(outerVariable)) {
                    return outer;
                }
                if (term.getKey().name().equals(innerVariable)) {
                    return inner;
                }
                return "static_cast<std::size_t>(" + term.getKey().name() + ")";
            }
        }
        StringBuilder result = new StringBuilder(Long.toString(expression.constant()));
        for (var term : expression.coefficients().entrySet()) {
            String variable = term.getKey().name().equals(outerVariable) ? outer
                : term.getKey().name().equals(innerVariable) ? inner : term.getKey().name();
            long coefficient = term.getValue();
            if (coefficient >= 0L) {
                result.append(" + ").append(coefficient).append(" * ").append(variable);
            } else {
                result.append(" - ").append(Math.abs(coefficient)).append(" * ").append(variable);
            }
        }
        return "static_cast<std::size_t>(" + result + ")";
    }

    static String functionSignature(KernelFunction function, String symbol) {
        StringBuilder result = new StringBuilder("extern \"C\" void ")
            .append(symbol).append('(');
        for (int index = 0; index < function.inputBuffers().size(); index++) {
            if (index > 0) {
                result.append(", ");
            }
            result.append("const double* b").append(index);
        }
        if (!function.inputBuffers().isEmpty()) {
            result.append(", ");
        }
        result.append("double* out, std::size_t rows, std::size_t cols)");
        return result.toString();
    }

    static void appendRegistryRegistration(StringBuilder source,
                                           PseudokernelPlan plan,
                                           CodegenBackend backend) {
        KernelFunction function = plan.function();
        source.append("\nextern \"C\" void ")
            .append(plan.signature().generatedSymbol()).append("_registry_entry(")
            .append("const double* const* inputs, std::size_t inputCount, double* out, ")
            .append("std::size_t rows, std::size_t cols) {\n")
            .append("    if (inputCount != ").append(function.inputBuffers().size())
            .append(") return;\n    ")
            .append(plan.signature().generatedSymbol()).append('(');
        for (int index = 0; index < function.inputBuffers().size(); index++) {
            if (index > 0) {
                source.append(", ");
            }
            source.append("inputs[").append(index).append(']');
        }
        if (!function.inputBuffers().isEmpty()) {
            source.append(", ");
        }
        source.append("out, rows, cols);\n}\n\n")
            .append("namespace {\nstruct ")
            .append(plan.signature().generatedSymbol()).append("_registrar {\n")
            .append("    ").append(plan.signature().generatedSymbol()).append("_registrar() {\n")
            .append("        const jlc_generated_kernel_descriptor descriptor{\n")
            .append("            \"").append(escape(plan.signature().canonicalText())).append("\",\n")
            .append("            \"").append(plan.signature().generatedSymbol()).append("\",\n")
            .append("            jlc_generated_backend::")
            .append(backend == CodegenBackend.AVX2 ? "AVX2" : "SCALAR_CPP").append(",\n")
            .append("            ").append(backend.vectorWidth()).append(",\n")
            .append("            \"FP64\",\n")
            .append("            ").append(function.loops().get(0).upperBound()
                - function.loops().get(0).lowerBound()).append(",\n")
            .append("            ").append(function.loops().get(1).upperBound()
                - function.loops().get(1).lowerBound()).append("\n        };\n")
            .append("        jlc_generated_register(descriptor, ")
            .append(plan.signature().generatedSymbol()).append("_registry_entry);\n")
            .append("    }\n};\n")
            .append("const ").append(plan.signature().generatedSymbol()).append("_registrar ")
            .append(plan.signature().generatedSymbol()).append("_registrar_instance{};\n}\n");
    }

    static String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n");
    }

    static boolean isConstant(KernelOp operation) {
        return operation.opcode() == KernelOpcode.CONSTANT;
    }
}
