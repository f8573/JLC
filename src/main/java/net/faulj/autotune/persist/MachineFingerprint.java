package net.faulj.autotune.persist;

import jdk.incubator.vector.DoubleVector;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

/**
 * Compute a stable machine fingerprint used to validate tuning profiles.
 */
public final class MachineFingerprint {

    private final String cpu;
    private final String os;
    private final String jvm;
    private final int vecLen;
    private final int processors;
    private final String hash;

    private MachineFingerprint(String cpu, String os, String jvm, int vecLen, int processors, String hash) {
        this.cpu = cpu;
        this.os = os;
        this.jvm = jvm;
        this.vecLen = vecLen;
        this.processors = processors;
        this.hash = hash;
    }

    public String getHash() { return hash; }
    public String getCpu() { return cpu; }
    public String getOs() { return os; }
    public String getJvm() { return jvm; }
    public int getVecLen() { return vecLen; }
    public int getProcessors() { return processors; }

    public static MachineFingerprint compute() {
        String cpu = detectCpuString();
        String os = String.format(Locale.ROOT, "%s/%s", System.getProperty("os.name", "unknown"), System.getProperty("os.arch", "unknown"));
        String jvm = String.format(Locale.ROOT, "%s|%s", System.getProperty("java.version", "?"), System.getProperty("java.vm.version", "?"));
        int vecLen = DoubleVector.SPECIES_PREFERRED.length();
        int procs = Runtime.getRuntime().availableProcessors();

        String blob = String.join("|", cpu, os, jvm, Integer.toString(vecLen), Integer.toString(procs));
        String hash = sha256Hex(blob);
        return new MachineFingerprint(cpu, os, jvm, vecLen, procs, hash);
    }

    private static String detectCpuString() {
        String envCpu = System.getenv("PROCESSOR_IDENTIFIER");
        if (envCpu != null && !envCpu.isEmpty()) return envCpu;
        envCpu = System.getenv("HOSTNAME");
        if (envCpu != null && !envCpu.isEmpty()) return envCpu;
        // Best-effort fallback
        return System.getProperty("os.arch", "unknown-cpu");
    }

    private static String sha256Hex(String s) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] dig = md.digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : dig) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }
}
