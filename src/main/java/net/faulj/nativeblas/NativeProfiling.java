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

    public static boolean setJniGemmArrayEnabled(boolean enabled) {
        try {
            NativeBindings.nativeJniGemmArrayProfileSetEnabled(enabled);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean resetJniGemmArray() {
        try {
            NativeBindings.nativeJniGemmArrayProfileReset();
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static Optional<NativeJniGemmArrayProfile> jniGemmArraySnapshot() {
        try {
            return Optional.of(NativeJniGemmArrayProfile.fromSnapshot(NativeBindings.nativeJniGemmArrayProfileSnapshot()));
        } catch (UnsatisfiedLinkError ignored) {
            return Optional.empty();
        }
    }

    public static boolean setQrEnabled(boolean enabled) {
        try {
            NativeBindings.nativeQrProfileSetEnabled(enabled);
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static boolean resetQr() {
        try {
            NativeBindings.nativeQrProfileReset();
            return true;
        } catch (UnsatisfiedLinkError ignored) {
            return false;
        }
    }

    public static Optional<NativeQrProfile> qrSnapshot() {
        try {
            return Optional.of(NativeQrProfile.fromSnapshot(NativeBindings.nativeQrProfileSnapshot()));
        } catch (UnsatisfiedLinkError ignored) {
            return Optional.empty();
        }
    }
}
