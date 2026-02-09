package net.faulj.bench;

public class TestPerfTimers {
    public static void main(String[] args) throws Exception {
        long t = net.faulj.util.PerfTimers.start();
        // small sleep to create measurable delta
        Thread.sleep(10);
        net.faulj.util.PerfTimers.record("TEST.PING", t);
        net.faulj.util.PerfTimers.dump(new java.io.File("build/reports/caqr_timers_test.csv"));
        System.out.println("Wrote test timers");
    }
}
