package net.faulj.compiler.matrix.codegen;

/** Explicit multiply/add contraction policy for a generated variant. */
public enum FmaContraction {
    OFF("off"),
    EXPLICIT("explicit");

    private final String propertyValue;

    FmaContraction(String propertyValue) {
        this.propertyValue = propertyValue;
    }

    public String propertyValue() {
        return propertyValue;
    }
}
