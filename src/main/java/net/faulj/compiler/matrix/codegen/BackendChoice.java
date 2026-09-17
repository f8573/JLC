package net.faulj.compiler.matrix.codegen;

import java.util.Optional;

/**
 * A runtime execution decision for one semantic kernel.
 *
 * <p>This type is deliberately separate from both {@link KernelSignature} and
 * {@link KernelVariantSignature}.  Java execution has no generated variant;
 * native choices carry the exact generated implementation that was validated
 * and, when calibrated, measured.</p>
 */
public sealed interface BackendChoice
    permits R2JavaBackend, ScalarNativeBackend, GeneratedAvx2Backend {

    /** Stable profile tag.  It is not used as the runtime dispatch type. */
    BackendKind kind();

    /** The exact generated implementation, when this choice has one. */
    default Optional<KernelVariantSignature> variantOptional() {
        return Optional.empty();
    }

    /** Stable serialization tag for profile files. */
    default String tag() {
        return kind().tag();
    }

    /** Runtime/backend choice tags used only at the profile boundary. */
    enum BackendKind {
        R2_JAVA("R2_JAVA"),
        SCALAR_NATIVE("SCALAR_NATIVE"),
        GENERATED_AVX2("GENERATED_AVX2");

        private final String tag;

        BackendKind(String tag) {
            this.tag = tag;
        }

        public String tag() {
            return tag;
        }

        public static BackendKind fromTag(String tag) {
            if (tag == null) {
                throw new IllegalArgumentException("Backend choice tag is missing");
            }
            for (BackendKind value : values()) {
                if (value.tag.equals(tag.trim())) {
                    return value;
                }
            }
            throw new IllegalArgumentException("Unknown backend choice tag: " + tag);
        }
    }
}
