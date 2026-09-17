package net.faulj.compiler.matrix.codegen;

/** Deterministic source-emission options for the R4 C++ backends. */
public record CppEmissionOptions(boolean nativeRegistryRegistration,
                                 String sourceProvenance) {
    public CppEmissionOptions {
        if (sourceProvenance == null || sourceProvenance.isBlank()) {
            sourceProvenance = "generated/r4/";
        }
    }

    public static CppEmissionOptions standalone() {
        return new CppEmissionOptions(false, "generated/r4/standalone");
    }

    public static CppEmissionOptions nativeRegistry() {
        return new CppEmissionOptions(true, "generated/r4/registry");
    }
}
