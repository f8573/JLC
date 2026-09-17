package net.faulj.compiler.matrix.codegen;

/** Developer selector for the optional generated-kernel execution path. */
public enum KernelBackendMode {
    R2("r2", null),
    SCALAR_CPP("scalar_cpp", CodegenBackend.SCALAR_CPP),
    AVX2("avx2", CodegenBackend.AVX2);

    public static final String PROPERTY = "jlc.compiler.kernelBackend";

    private final String propertyValue;
    private final CodegenBackend backend;

    KernelBackendMode(String propertyValue, CodegenBackend backend) {
        this.propertyValue = propertyValue;
        this.backend = backend;
    }

    public String propertyValue() {
        return propertyValue;
    }

    public CodegenBackend backend() {
        return backend;
    }

    public boolean isGenerated() {
        return backend != null;
    }

    public static KernelBackendMode fromSystemProperty() {
        String configured = System.getProperty(PROPERTY);
        if (configured == null || configured.isBlank()) {
            return R2;
        }
        for (KernelBackendMode mode : values()) {
            if (mode.propertyValue.equalsIgnoreCase(configured.trim())) {
                return mode;
            }
        }
        throw new IllegalArgumentException(
            "Unsupported " + PROPERTY + " value '" + configured
                + "'; expected r2, scalar_cpp, or avx2");
    }
}
