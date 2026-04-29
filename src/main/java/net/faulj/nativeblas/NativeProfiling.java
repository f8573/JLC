package net.faulj.nativeblas;

import java.util.Optional;

/**
 * Safe Java access to the optional native GEMM profiler.
 */
public final class NativeProfiling {
    private NativeProfiling() {
    }

    public static boolean setEnabled(boolean enabled) {
        try {
            NativeBindings.nativeProfileSetEnabled(enabled);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean reset() {
        try {
            NativeBindings.nativeProfileReset();
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static Optional<NativeGemmProfile> snapshot() {
        try {
            return Optional.of(NativeGemmProfile.fromSnapshot(NativeBindings.nativeProfileSnapshot()));
        } catch (UnsatisfiedLinkError ignored) {
            return Optional.empty();
        }
    }
}
