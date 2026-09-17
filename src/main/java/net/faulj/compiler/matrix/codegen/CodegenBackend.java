package net.faulj.compiler.matrix.codegen;

/** Native source backends currently supported by R4. */
public enum CodegenBackend {
    SCALAR_CPP("scalar_cpp", "none", 1),
    AVX2("avx2", "avx2", 4);

    private final String propertyValue;
    private final String requiredCpuFeature;
    private final int vectorWidth;

    CodegenBackend(String propertyValue, String requiredCpuFeature, int vectorWidth) {
        this.propertyValue = propertyValue;
        this.requiredCpuFeature = requiredCpuFeature;
        this.vectorWidth = vectorWidth;
    }

    public String propertyValue() {
        return propertyValue;
    }

    public String requiredCpuFeature() {
        return requiredCpuFeature;
    }

    public int vectorWidth() {
        return vectorWidth;
    }
}
