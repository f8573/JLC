package net.faulj.compiler.matrix.codegen;

/** Tail handling policy for generated vector loops. */
public enum KernelTailPolicy {
    SCALAR("scalar");

    private final String propertyValue;

    KernelTailPolicy(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }
}
