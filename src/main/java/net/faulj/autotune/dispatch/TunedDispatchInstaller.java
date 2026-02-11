package net.faulj.autotune.dispatch;

import net.faulj.compute.DispatchPolicy;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Install tuned block sizes into the global dispatch policy.
 * Installation is one-time and lock-free at call time.
 */
public final class TunedDispatchInstaller {

    private static volatile BlockSizes activeBlockSizes;
    private static final AtomicBoolean installed = new AtomicBoolean(false);

    private TunedDispatchInstaller() {}

    public static BlockSizes getActiveBlockSizes() {
        return activeBlockSizes;
    }

    public static void install(BlockSizes sizes) {
        if (sizes == null) return;
        if (!installed.compareAndSet(false, true)) {
            // already installed; idempotent
            return;
        }
        activeBlockSizes = sizes;

        // Create a DispatchPolicy that uses tuned MC as block size base.
        try {
            DispatchPolicy.Builder b = DispatchPolicy.builder();
            int bs = Math.max(1, sizes.mc);
            b.blockSize(bs).minBlockSize(bs).maxBlockSize(bs);
            DispatchPolicy policy = b.build();
            DispatchPolicy.setGlobalPolicy(policy);
            System.out.printf("Installed tuned dispatch (MC=%d, KC=%d, NC=%d)%n", sizes.mc, sizes.kc, sizes.nc);
        } catch (Throwable t) {
            System.err.println("Failed to install tuned dispatch: " + t.getMessage());
        }
    }
}
