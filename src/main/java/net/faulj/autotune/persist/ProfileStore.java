package net.faulj.autotune.persist;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Optional;

/**
 * Persist and load tuning profiles under ~/.jlc/tuning/ using a small, dependency-free JSON helper.
 */
public final class ProfileStore {

    private static final String BASE_DIR = System.getProperty("user.home") + File.separator + ".jlc" + File.separator + "tuning";
    private static final String SCHEMA = "jlc.gemm.profile.v1";
    private static final String PIPELINE = "autotune-v1";

    private ProfileStore() {}

    public static Path save(TuningProfile profile) {
        try {
            Path dir = Paths.get(BASE_DIR);
            Files.createDirectories(dir);
            String filename = String.format("gemm-%s.json", profile.fingerprint);
            Path out = dir.resolve(filename);
            String json = toJson(profile);
            Files.write(out, json.getBytes(StandardCharsets.UTF_8));
            return out;
        } catch (IOException e) {
            System.err.println("Failed to persist tuning profile: " + e.getMessage());
            return null;
        }
    }

    public static Optional<TuningProfile> loadByFingerprint(String fingerprint) {
        Path p = Paths.get(BASE_DIR, String.format("gemm-%s.json", fingerprint));
        if (!Files.exists(p)) return Optional.empty();
        try {
            String json = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
            TuningProfile prof = fromJson(json);
            return Optional.ofNullable(prof);
        } catch (IOException e) {
            System.err.println("Failed to load tuning profile: " + e.getMessage());
            return Optional.empty();
        }
    }

    public static Optional<TuningProfile> loadAny() {
        try {
            Path dir = Paths.get(BASE_DIR);
            if (!Files.exists(dir)) return Optional.empty();
            try (java.util.stream.Stream<Path> ps = Files.list(dir)) {
                for (Path p : (Iterable<Path>) ps::iterator) {
                    String name = p.getFileName().toString();
                    if (!name.startsWith("gemm-") || !name.endsWith(".json")) continue;
                    try {
                        String json = new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
                        TuningProfile prof = fromJson(json);
                        if (prof != null) return Optional.of(prof);
                    } catch (IOException ignore) {
                    }
                }
            }
        } catch (IOException ignore) {
        }
        return Optional.empty();
    }

    public static TuningProfile createFrom(String fingerprint, boolean converged, String terminationReason,
                                           int mr, int nr, int kc, int kUnroll, int mc, int nc,
                                           double gflops, double cv) {
        return new TuningProfile(fingerprint, Instant.now().toString(), converged, terminationReason,
                mr, nr, kc, kUnroll, mc, nc, gflops, cv, SCHEMA, PIPELINE);
    }

    /**
     * Create a tuning profile directly from a Phase 5 ConvergenceReport.
     * This preserves Phase 5 truth for `converged` and `terminationReason` without
     * re-deriving or normalizing those fields elsewhere.
     */
    public static TuningProfile createFrom(String fingerprint, net.faulj.autotune.converge.ConvergenceReport report) {
        if (report == null) {
            return new TuningProfile(fingerprint, Instant.now().toString(), false, "",
                    0, 0, 0, 0, 0, 0, 0.0, 0.0, SCHEMA, PIPELINE);
        }
        net.faulj.autotune.benchmark.FineSearchResult sel = report.selected;
        if (sel == null) {
            return new TuningProfile(fingerprint, Instant.now().toString(), report.converged, report.reason,
                    0, 0, 0, 0, 0, 0, 0.0, 0.0, SCHEMA, PIPELINE);
        }
        return new TuningProfile(fingerprint, Instant.now().toString(), report.converged, report.reason,
                sel.mr, sel.nr, sel.bestKc, sel.bestKUnroll, sel.mc, sel.nc,
                sel.bestGflops, sel.cv, SCHEMA, PIPELINE);
    }

    // Minimal JSON serialization (no escaping needed for our controlled fields)
    private static String toJson(TuningProfile p) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        appendKV(sb, "fingerprint", p.fingerprint).append(',');
        appendKV(sb, "timestamp", p.timestamp).append(',');
        appendKV(sb, "converged", p.converged).append(',');
        appendKV(sb, "terminationReason", p.terminationReason).append(',');
        appendKV(sb, "mr", p.mr).append(',');
        appendKV(sb, "nr", p.nr).append(',');
        appendKV(sb, "kc", p.kc).append(',');
        appendKV(sb, "kUnroll", p.kUnroll).append(',');
        appendKV(sb, "mc", p.mc).append(',');
        appendKV(sb, "nc", p.nc).append(',');
        appendKV(sb, "gflops", p.gflops).append(',');
        appendKV(sb, "cv", p.cv).append(',');
        appendKV(sb, "schemaVersion", p.schemaVersion).append(',');
        appendKV(sb, "pipelineVersion", p.pipelineVersion);
        sb.append('}');
        return sb.toString();
    }

    private static StringBuilder appendKV(StringBuilder sb, String k, String v) {
        sb.append('"').append(k).append('"').append(':');
        sb.append('"').append(v == null ? "" : v).append('"');
        return sb;
    }

    private static StringBuilder appendKV(StringBuilder sb, String k, boolean v) {
        sb.append('"').append(k).append('"').append(':').append(v);
        return sb;
    }

    private static StringBuilder appendKV(StringBuilder sb, String k, int v) {
        sb.append('"').append(k).append('"').append(':').append(v);
        return sb;
    }

    private static StringBuilder appendKV(StringBuilder sb, String k, double v) {
        sb.append('"').append(k).append('"').append(':').append(v);
        return sb;
    }

    // Minimal parser for the exact schema we write (robust to spacing)
    private static TuningProfile fromJson(String s) {
        try {
            java.util.Map<String, String> map = new java.util.HashMap<>();
            String t = s.trim();
            if (t.startsWith("{")) t = t.substring(1);
            if (t.endsWith("}")) t = t.substring(0, t.length() - 1);
            String[] parts = t.split(",(?=\")");
            for (String part : parts) {
                String[] kv = part.split(":", 2);
                if (kv.length != 2) continue;
                String key = kv[0].trim();
                if (key.startsWith("\"")) key = key.substring(1);
                if (key.endsWith("\"")) key = key.substring(0, key.length() - 1);
                String val = kv[1].trim();
                if (val.startsWith("\"")) val = val.substring(1);
                if (val.endsWith("\"")) val = val.substring(0, val.length() - 1);
                map.put(key, val);
            }
            String fingerprint = map.getOrDefault("fingerprint", "");
            String timestamp = map.getOrDefault("timestamp", "");
            boolean converged = Boolean.parseBoolean(map.getOrDefault("converged", "false"));
            String terminationReason = map.getOrDefault("terminationReason", "");
            int mr = Integer.parseInt(map.getOrDefault("mr", "0"));
            int nr = Integer.parseInt(map.getOrDefault("nr", "0"));
            int kc = Integer.parseInt(map.getOrDefault("kc", "0"));
            int kUnroll = Integer.parseInt(map.getOrDefault("kUnroll", "0"));
            int mc = Integer.parseInt(map.getOrDefault("mc", "0"));
            int nc = Integer.parseInt(map.getOrDefault("nc", "0"));
            double gflops = Double.parseDouble(map.getOrDefault("gflops", "0"));
            double cv = Double.parseDouble(map.getOrDefault("cv", "0"));
            String schema = map.getOrDefault("schemaVersion", "");
            String pipeline = map.getOrDefault("pipelineVersion", "");
            return new TuningProfile(fingerprint, timestamp, converged, terminationReason,
                    mr, nr, kc, kUnroll, mc, nc, gflops, cv, schema, pipeline);
        } catch (Throwable t) {
            System.err.println("Failed to parse tuning profile JSON: " + t.getMessage());
            return null;
        }
    }
}
