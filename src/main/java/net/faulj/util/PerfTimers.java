package net.faulj.util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Very small runtime profiler for collecting per-method elapsed-time samples.
 * Records total nanoseconds and invocation counts per tag and writes a CSV
 * to build/reports/caqr_timers.csv on JVM shutdown.
 */
public final class PerfTimers {
    private static final Map<String, LongAdder> COUNTS = new ConcurrentHashMap<>();
    private static final Map<String, LongAdder> NANOS = new ConcurrentHashMap<>();

    static {
        // Dump on exit
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                dump(new File("build/reports/caqr_timers.csv"));
            } catch (IOException e) {
                // best-effort
                e.printStackTrace();
            }
        }));
    }

    private PerfTimers() {}

    public static long start() {
        return System.nanoTime();
    }

    public static void record(String tag, long startNanos) {
        long delta = System.nanoTime() - startNanos;
        NANOS.computeIfAbsent(tag, k -> new LongAdder()).add(delta);
        COUNTS.computeIfAbsent(tag, k -> new LongAdder()).increment();
    }

    public static void dump(File out) throws IOException {
        File parent = out.getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();

        try (FileWriter fw = new FileWriter(out)) {
            fw.write("tag,invocations,total_nanos,avg_nanos\n");
            for (Map.Entry<String, LongAdder> e : NANOS.entrySet()) {
                String tag = e.getKey();
                long total = e.getValue().sum();
                long inv = COUNTS.getOrDefault(tag, new LongAdder()).sum();
                long avg = inv == 0 ? 0 : total / inv;
                fw.write(tag + "," + inv + "," + total + "," + avg + "\n");
            }
        }
    }
}
