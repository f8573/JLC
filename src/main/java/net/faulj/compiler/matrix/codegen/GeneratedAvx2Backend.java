package net.faulj.compiler.matrix.codegen;

import java.util.Objects;
import java.util.Optional;

/** An exact generated AVX2 implementation invoked through JNI. */
public record GeneratedAvx2Backend(KernelVariantSignature variant) implements BackendChoice {
    public GeneratedAvx2Backend {
        Objects.requireNonNull(variant, "Generated AVX2 variant");
        if (variant.backend() != CodegenBackend.AVX2) {
            throw new IllegalArgumentException("Generated AVX2 choices require an AVX2 variant");
        }
    }

    @Override
    public BackendKind kind() {
        return BackendKind.GENERATED_AVX2;
    }

    @Override
    public Optional<KernelVariantSignature> variantOptional() {
        return Optional.of(variant);
    }
}
