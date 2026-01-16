package net.faulj.core;

/**
 * Global epsilon/tolerance strategy for numerical comparisons.
 * Thread-safe singleton pattern with configurable default.
 */
public final class Tolerance {
    
    private static volatile double defaultTolerance = 1e-10;
    
    private Tolerance() {}
    
    public static double get() {
        return defaultTolerance;
    }
    
    public static void set(double tol) {
        if (tol < 0) {
            throw new IllegalArgumentException("Tolerance must be non-negative");
        }
        defaultTolerance = tol;
    }
    
    public static boolean isZero(double value) {
        return Math.abs(value) <= defaultTolerance;
    }
    
    public static boolean isZero(double value, double tol) {
        return Math.abs(value) <= tol;
    }
    
    public static boolean equals(double a, double b) {
        return Math.abs(a - b) <= defaultTolerance;
    }
    
    public static boolean equals(double a, double b, double tol) {
        return Math.abs(a - b) <= tol;
    }
}
