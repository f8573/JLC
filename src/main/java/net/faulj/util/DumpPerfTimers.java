package net.faulj.util;

import java.io.File;
import java.io.IOException;

public class DumpPerfTimers {
    public static void main(String[] args) {
        try {
            PerfTimers.dump(new File("build/reports/caqr_timers.csv"));
            System.out.println("Wrote build/reports/caqr_timers.csv");
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
