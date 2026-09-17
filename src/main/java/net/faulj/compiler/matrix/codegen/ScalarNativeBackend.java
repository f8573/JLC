package net.faulj.compiler.matrix.codegen;

import java.util.Objects;
import java.util.Optional;

/** An exact generated scalar C++ implementation invoked through JNI. */
public record ScalarNativeBackend(KernelVariantSignature variant) implements BackendChoice {
    public ScalarNativeBackend {
        Objects.requireNonNull(variant, "Scalar-native variant");
        if (variant.backend() != CodegenBackend.SCALAR_CPP) {
            throw new IllegalArgumentException("Scalar-native choices require a scalar variant");
        }
    }

    @Override
    public BackendKind kind() {
        return BackendKind.SCALAR_NATIVE;
    }

    @Override
    public Optional<KernelVariantSignature> variantOptional() {
        return Optional.of(variant);
    }
}
