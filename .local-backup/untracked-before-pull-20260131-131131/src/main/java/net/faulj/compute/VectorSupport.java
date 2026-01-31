package net.faulj.compute;

/**
 * Helper for detecting availability of the Java Vector API at runtime.
 * <p>
 * Uses reflection to probe for known Vector API classes present in different
 * JDK module/package names across Java releases.
 * </p>
 */
public final class VectorSupport {
    private VectorSupport() {
        // utility
    }

    /**
     * Returns true if a Vector API implementation appears available on the runtime.
     *
     * @return true when Vector API classes can be loaded, false otherwise
     */
    public static boolean isVectorApiAvailable() {
        try {
            Class.forName("jdk.incubator.vector.VectorSpecies");
            return true;
        } catch (ClassNotFoundException ignored) {
        }

        try {
            Class.forName("jdk.vector.VectorSpecies");
            return true;
        } catch (ClassNotFoundException ignored) {
        }

        return false;
    }
}
