package net.faulj.web;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Global compute governor enforcing a hard concurrency cap and providing
 * simple per-request thread allocation heuristics.
 */
public class ComputeGovernor {
    private final int maxParallelJobs;
    private final Semaphore semaphore;
    private final AtomicInteger activeJobs = new AtomicInteger(0);
    private final AtomicInteger acquiredCount = new AtomicInteger(0);
    private final AtomicInteger rejectedCount = new AtomicInteger(0);
    private final AtomicInteger resetCount = new AtomicInteger(0);

    // Track acquisition timestamps to detect stuck/long-running acquisitions
    private final ConcurrentLinkedQueue<Long> acquireTimestamps = new ConcurrentLinkedQueue<>();

    // Watchdog to detect stuck state and optionally auto-recover
    private final ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "ComputeGovernor-Watchdog");
        t.setDaemon(true);
        return t;
    });

    public ComputeGovernor() {
        this(Runtime.getRuntime().availableProcessors());
    }

    public ComputeGovernor(int maxParallelJobs) {
        this.maxParallelJobs = Math.max(1, maxParallelJobs);
        this.semaphore = new Semaphore(this.maxParallelJobs, true);
        // Start periodic watchdog to detect stuck acquisitions
        long checkIntervalSec = 60L;
        long thresholdMinutes = Math.max(1, parseLong(System.getProperty("faulj.governor.stuckThresholdMinutes"), 30L));
        long thresholdMs = TimeUnit.MINUTES.toMillis(thresholdMinutes);
        watchdog.scheduleAtFixedRate(() -> {
            try {
                int active = getActiveJobs();
                Long oldest = acquireTimestamps.peek();
                long now = System.currentTimeMillis();
                boolean tooMany = active > maxParallelJobs;
                boolean stale = oldest != null && (now - oldest) > thresholdMs;
                if (tooMany || stale) {
                    System.err.printf("[GOVERNOR WATCHDOG] detected stuck state: active=%d max=%d oldestAgeMs=%d tooMany=%b stale=%b\n",
                            active, maxParallelJobs, oldest == null ? 0L : (now - oldest), tooMany, stale);
                    // attempt a safe reset
                    forceReset();
                    resetCount.incrementAndGet();
                }
            } catch (Throwable t) {
                System.err.println("[GOVERNOR WATCHDOG] error: " + t.getMessage());
            }
        }, checkIntervalSec, checkIntervalSec, TimeUnit.SECONDS);
    }

    /** Try to acquire a compute slot. Returns true when caller may proceed. */
    public boolean tryAcquire() {
        boolean ok = semaphore.tryAcquire();
        if (ok) {
            activeJobs.incrementAndGet();
            acquiredCount.incrementAndGet();
            acquireTimestamps.add(System.currentTimeMillis());
        }
        else {
            rejectedCount.incrementAndGet();
        }
        return ok;
    }

    /** Release a previously acquired compute slot. */
    public void release() {
        int v = activeJobs.updateAndGet(x -> Math.max(0, x - 1));
        // remove one timestamp corresponding to this release (best-effort)
        try { acquireTimestamps.poll(); } catch (Throwable ignored) {}
        try {
            semaphore.release();
        } catch (Throwable ignored) {}
    }

    /**
     * Administrative: force-reset the governor state.
     * Sets activeJobs to zero and restores semaphore permits to maxParallelJobs.
     * Use only for troubleshooting/stuck-state recovery.
     */
    public synchronized void forceReset() {
        activeJobs.set(0);
        try { acquireTimestamps.clear(); } catch (Throwable ignored) {}
        try {
            int avail = semaphore.availablePermits();
            int toRelease = Math.max(0, maxParallelJobs - avail);
            if (toRelease > 0) semaphore.release(toRelease);
        } catch (Throwable ignored) {}
    }

    private static long parseLong(String s, long fallback) {
        if (s == null) return fallback;
        try { return Long.parseLong(s.trim()); } catch (Exception ex) { return fallback; }
    }

    /** Shutdown watchdog thread pool (call on app shutdown). */
    public void shutdownWatchdog() {
        try {
            watchdog.shutdownNow();
        } catch (Throwable ignored) {}
    }

    public int getAcquiredCount() { return acquiredCount.get(); }
    public int getRejectedCount() { return rejectedCount.get(); }
    public int getResetCount() { return resetCount.get(); }

    /** Age of oldest acquisition in seconds (0 if none). */
    public long getOldestAcquireAgeSeconds() {
        Long oldest = acquireTimestamps.peek();
        if (oldest == null) return 0L;
        long ageMs = System.currentTimeMillis() - oldest;
        return TimeUnit.MILLISECONDS.toSeconds(Math.max(0, ageMs));
    }

    /** Current number of active jobs. */
    public int getActiveJobs() {
        return activeJobs.get();
    }

    /** Maximum parallel jobs allowed. */
    public int getMaxParallelJobs() {
        return maxParallelJobs;
    }

    /**
     * Compute per-job thread allotment using floor(totalCores / activeJobs).
     * Ensures at least 1 thread per job.
     */
    public int computeThreadsPerJob() {
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors());
        int jobs = Math.max(1, getActiveJobs());
        return Math.max(1, cores / jobs);
    }
}
