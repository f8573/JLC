package net.faulj.compiler.matrix.codegen;

import java.util.Objects;

import net.faulj.nativeblas.NativeGeneratedKernelSupport;

/**
 * Immutable facts used to validate a cached backend choice.
 *
 * <p>Construction is the runtime boundary at which native loading and ISA
 * probing are allowed.  Kernel invocation only consumes this snapshot.</p>
 */
public final class RuntimeEnvironment {
    private final GeneratedKernelRegistry registry;
    private final KernelMachineIdentity machine;
    private final KernelBuildIdentity build;
    private final boolean javaExecutorAvailable;
    private final boolean nativeRuntimeAvailable;
    private final boolean avx2Supported;
    private final boolean fmaSupported;

    public RuntimeEnvironment(GeneratedKernelRegistry registry,
                              KernelMachineIdentity machine,
                              KernelBuildIdentity build,
                              boolean javaExecutorAvailable,
                              boolean nativeRuntimeAvailable,
                              boolean avx2Supported) {
        this(registry, machine, build, javaExecutorAvailable, nativeRuntimeAvailable,
            avx2Supported, RuntimeCpuFeatures.fmaSupported());
    }

    public RuntimeEnvironment(GeneratedKernelRegistry registry,
                              KernelMachineIdentity machine,
                              KernelBuildIdentity build,
                              boolean javaExecutorAvailable,
                              boolean nativeRuntimeAvailable,
                              boolean avx2Supported,
                              boolean fmaSupported) {
        this.registry = registry == null ? GeneratedKernelRegistry.global() : registry;
        this.machine = Objects.requireNonNull(machine, "Machine identity");
        this.build = Objects.requireNonNull(build, "Build identity");
        this.javaExecutorAvailable = javaExecutorAvailable;
        this.nativeRuntimeAvailable = nativeRuntimeAvailable;
        this.avx2Supported = avx2Supported;
        this.fmaSupported = fmaSupported;
    }

    /** Capture the production runtime facts once. */
    public static RuntimeEnvironment current(GeneratedKernelRegistry registry,
                                             KernelMachineIdentity machine,
                                             KernelBuildIdentity build) {
        return new RuntimeEnvironment(
            registry,
            machine == null ? KernelMachineIdentity.current() : machine,
            build == null ? KernelBuildIdentity.current() : build,
            true,
            NativeGeneratedKernelSupport.isAvailable(),
            RuntimeCpuFeatures.avx2Supported(),
            RuntimeCpuFeatures.fmaSupported());
    }

    public static RuntimeEnvironment current() {
        return current(GeneratedKernelRegistry.global(), null, null);
    }

    public GeneratedKernelRegistry registry() {
        return registry;
    }

    public KernelMachineIdentity machine() {
        return machine;
    }

    public KernelBuildIdentity build() {
        return build;
    }

    public boolean javaExecutorAvailable() {
        return javaExecutorAvailable;
    }

    public boolean nativeRuntimeAvailable() {
        return nativeRuntimeAvailable;
    }

    public boolean avx2Supported() {
        return avx2Supported;
    }

    public boolean fmaSupported() {
        return fmaSupported;
    }

    /** Return an empty string when the choice is safe to invoke. */
    public String validationReason(KernelSignature signature, BackendChoice choice) {
        if (signature == null) return "kernel signature is null";
        if (choice == null) return "backend choice is null";
        if (choice instanceof R2JavaBackend) {
            return javaExecutorAvailable ? "available" : "Java executor unavailable";
        }
        if (!nativeRuntimeAvailable) {
            return "native runtime unavailable";
        }
        KernelVariantSignature variant = choice.variantOptional().orElse(null);
        if (variant == null) {
            return "native choice has no variant";
        }
        if (!signature.equals(variant.kernelSignature())) {
            return "variant semantic signature mismatch";
        }
        GeneratedKernelRegistry.Entry entry = registry.lookup(variant);
        if (entry == null) {
            return "generated artifact is not registered";
        }
        if (choice instanceof ScalarNativeBackend) {
            return variant.backend() == CodegenBackend.SCALAR_CPP
                ? "available" : "scalar choice has non-scalar variant";
        }
        if (!(choice instanceof GeneratedAvx2Backend)) {
            return "unknown native backend choice";
        }
        if (!avx2Supported) {
            return "AVX2 is unavailable";
        }
        if (variant.backend() != CodegenBackend.AVX2) {
            return "AVX2 choice has non-AVX2 variant";
        }
        if ("avx2+fma".equals(variant.requiredCpuFeature())
            && !fmaSupported) {
            return "required FMA ISA is unavailable";
        }
        if (variant.fmaContraction() == FmaContraction.EXPLICIT
            && !fmaSupported) {
            return "required FMA ISA is unavailable";
        }
        return "available";
    }

    public boolean canExecute(KernelSignature signature, BackendChoice choice) {
        return "available".equals(validationReason(signature, choice));
    }
}
