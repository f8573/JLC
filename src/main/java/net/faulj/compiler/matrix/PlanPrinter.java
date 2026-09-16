package net.faulj.compiler.matrix;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Deterministic textual renderings for plan inspection and tests.
 */
final class PlanPrinter {
    private PlanPrinter() {
    }

    static String expression(PlanNode root) {
        PlanNode.requireNode(root, "Plan root");
        return new ExpressionPrinter().format(root, 0);
    }

    static String dump(PlanNode root) {
        PlanNode.requireNode(root, "Plan root");
        SsaPrinter printer = new SsaPrinter();
        printer.visit(root);
        StringBuilder result = new StringBuilder();
        for (String line : printer.lines) {
            if (result.length() > 0) {
                result.append('\n');
            }
            result.append(line);
        }
        result.append('\n')
            .append("result = %")
            .append(printer.ids.get(root))
            .append(" : ")
            .append(root.shape());
        return result.toString();
    }

    private static String indent(int count) {
        return " ".repeat(count);
    }

    private static String displayName(PlanInput input, IdentityHashMap<PlanInput, Integer> inputIds) {
        String name = input.name();
        if (name != null) {
            return name;
        }
        Integer id = inputIds.get(input);
        if (id == null) {
            id = inputIds.size();
            inputIds.put(input, id);
        }
        return "%" + id;
    }

    private static String safeName(String name) {
        return name.replace('\n', ' ').replace('\r', ' ');
    }

    private static final class ExpressionPrinter {
        private final IdentityHashMap<PlanInput, Integer> inputIds = new IdentityHashMap<>();

        private String format(PlanNode node, int indentation) {
            if (node instanceof PlanInput input) {
                return indent(indentation) + "input(" + safeName(displayName(input, inputIds)) + ")";
            }
            if (node instanceof PlanMatMul matMul) {
                return binary("matmul", matMul.lhs(), matMul.rhs(), indentation);
            }
            if (node instanceof PlanAdd add) {
                return binary("add", add.lhs(), add.rhs(), indentation);
            }
            if (node instanceof PlanScale scale) {
                return indent(indentation) + "scale(" + Double.toString(scale.factor()) + ",\n"
                    + format(scale.operand(), indentation + 2) + "\n"
                    + indent(indentation) + ")";
            }
            PlanTranspose transpose = (PlanTranspose) node;
            return indent(indentation) + "transpose(\n"
                + format(transpose.operand(), indentation + 2) + "\n"
                + indent(indentation) + ")";
        }

        private String binary(String operation, PlanNode left, PlanNode right, int indentation) {
            return indent(indentation) + operation + "(\n"
                + format(left, indentation + 2) + ",\n"
                + format(right, indentation + 2) + "\n"
                + indent(indentation) + ")";
        }
    }

    private static final class SsaPrinter {
        private final IdentityHashMap<PlanNode, Integer> ids = new IdentityHashMap<>();
        private final List<String> lines = new ArrayList<>();

        private void visit(PlanNode node) {
            if (ids.containsKey(node)) {
                return;
            }
            if (node instanceof PlanMatMul matMul) {
                visit(matMul.lhs());
                visit(matMul.rhs());
            } else if (node instanceof PlanAdd add) {
                visit(add.lhs());
                visit(add.rhs());
            } else if (node instanceof PlanScale scale) {
                visit(scale.operand());
            } else if (node instanceof PlanTranspose transpose) {
                visit(transpose.operand());
            }

            int id = ids.size();
            ids.put(node, id);
            lines.add(formatDefinition(node, id));
        }

        private String formatDefinition(PlanNode node, int id) {
            String prefix = "%" + id + " = ";
            if (node instanceof PlanInput input) {
                String operation = input.source() instanceof SymbolicInput ? "symbolic_input" : "input";
                String name = input.name();
                return prefix + operation + (name == null ? "" : " " + safeName(name))
                    + " : " + node.shape();
            }
            if (node instanceof PlanMatMul matMul) {
                return prefix + "matmul %" + ids.get(matMul.lhs()) + ", %" + ids.get(matMul.rhs())
                    + " : " + node.shape();
            }
            if (node instanceof PlanAdd add) {
                return prefix + "add %" + ids.get(add.lhs()) + ", %" + ids.get(add.rhs())
                    + " : " + node.shape();
            }
            if (node instanceof PlanScale scale) {
                return prefix + "scale " + Double.toString(scale.factor()) + ", %"
                    + ids.get(scale.operand()) + " : " + node.shape();
            }
            PlanTranspose transpose = (PlanTranspose) node;
            return prefix + "transpose %" + ids.get(transpose.operand()) + " : " + node.shape();
        }
    }
}
