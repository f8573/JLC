package net.faulj.compiler.matrix.codegen;

/** Backend traversal form for an already legal R3 kernel. */
public enum KernelLoopForm {
    FLAT("flat"),
    NESTED("nested");

    private final String propertyValue;

    KernelLoopForm(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }
}
