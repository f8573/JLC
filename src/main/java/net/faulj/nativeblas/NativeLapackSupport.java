package net.faulj.nativeblas;

final class NativeLapackSupport {
    private NativeLapackSupport() {
    }

    static boolean shouldUseVendorLapack(String algorithmKey, int problemSize, int defaultMinSize) {
        if (problemSize < minimumSize(algorithmKey, defaultMinSize)) {
            return false;
        }
        if (!providerAllowsVendorLapack(algorithmKey)) {
            return false;
        }
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        if (!"native".equals(snapshot.activeBackend()) || !snapshot.nativeContext().isAvailable()) {
            return false;
        }
        return NativeBindings.nativeVendorLapackAvailable();
    }

    private static boolean providerAllowsVendorLapack(String algorithmKey) {
        String configured = firstNonBlank(
            System.getProperty("jlc.native." + algorithmKey + ".provider"),
            System.getProperty("faulj.native." + algorithmKey + ".provider"),
            System.getProperty("jlc.native.provider"),
            System.getProperty("faulj.native.provider")
        );
        if (configured == null || configured.isBlank()) {
            return true;
        }
        return switch (configured.trim().toLowerCase()) {
            case "java", "disabled", "builtin", "native", "kernel" -> false;
            default -> true;
        };
    }

    private static int minimumSize(String algorithmKey, int defaultValue) {
        String configured = firstNonBlank(
            System.getProperty("jlc.native." + algorithmKey + ".minSize"),
            System.getProperty("faulj.native." + algorithmKey + ".minSize")
        );
        if (configured == null || configured.isBlank()) {
            return defaultValue;
        }
        try {
            return Math.max(1, Integer.parseInt(configured.trim()));
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }
}
