package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

/**
 * Human-inspectable machine/build-specific R5 evidence profile.
 *
 * <p>The profile is evidence, not just a winner cache: pruned, failed, noisy,
 * and losing candidates remain serialized with raw samples where available.</p>
 */
public final class KernelCalibrationProfile {
    public static final int CURRENT_SCHEMA_VERSION = 1;
    public static final String METHODOLOGY_VERSION = "r5-median-mad-v1";

    private static final ObjectMapper MAPPER = new ObjectMapper()
        .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    private final int schemaVersion;
    private final KernelMachineIdentity machine;
    private final KernelBuildIdentity build;
    private final String timestamp;
    private final String methodologyVersion;
    private final List<KernelCalibrationEntry> entries;

    public KernelCalibrationProfile(KernelMachineIdentity machine,
                                    KernelBuildIdentity build,
                                    String timestamp,
                                    String methodologyVersion,
                                    List<KernelCalibrationEntry> entries) {
        this(CURRENT_SCHEMA_VERSION, machine, build, timestamp, methodologyVersion, entries);
    }

    private KernelCalibrationProfile(int schemaVersion,
                                     KernelMachineIdentity machine,
                                     KernelBuildIdentity build,
                                     String timestamp,
                                     String methodologyVersion,
                                     List<KernelCalibrationEntry> entries) {
        this.schemaVersion = schemaVersion;
        this.machine = machine;
        this.build = build;
        this.timestamp = timestamp == null ? "" : timestamp;
        this.methodologyVersion = methodologyVersion == null ? METHODOLOGY_VERSION
            : methodologyVersion;
        this.entries = entries == null ? List.of() : List.copyOf(entries);
    }

    public static KernelCalibrationProfile fromResult(KernelMachineIdentity machine,
                                                      KernelBuildIdentity build,
                                                      KernelTuningResult result) {
        if (machine == null || build == null || result == null) {
            throw new IllegalArgumentException("Machine, build, and tuning result are required");
        }
        return new KernelCalibrationProfile(machine, build,
            java.time.Instant.now().toString(), METHODOLOGY_VERSION,
            List.of(KernelCalibrationEntry.from(result)));
    }

    public int schemaVersion() { return schemaVersion; }
    public KernelMachineIdentity machine() { return machine; }
    public KernelBuildIdentity build() { return build; }
    public String machineKey() { return machine == null ? "" : machine.key(); }
    public String buildKey() { return build == null ? "" : build.key(); }
    public String timestamp() { return timestamp; }
    public String methodologyVersion() { return methodologyVersion; }
    public List<KernelCalibrationEntry> entries() { return entries; }

    public Optional<KernelCalibrationEntry> entry(KernelSignature signature) {
        if (signature == null) return Optional.empty();
        return entries.stream().filter(entry -> signature.sha256().equals(entry.kernelSha256())
            && signature.canonicalText().equals(entry.kernelSignature())).findFirst();
    }

    public String compatibilityReason(KernelMachineIdentity currentMachine,
                                      KernelBuildIdentity currentBuild,
                                      KernelSignature signature) {
        if (schemaVersion != CURRENT_SCHEMA_VERSION) return "schema version mismatch";
        if (machine == null || currentMachine == null || !machine.key().equals(currentMachine.key())) {
            return "machine mismatch";
        }
        if (build == null || currentBuild == null || !build.key().equals(currentBuild.key())) {
            return "build mismatch";
        }
        if (entry(signature).isEmpty()) return "kernel signature missing";
        return "matched";
    }

    public boolean compatibleWith(KernelMachineIdentity currentMachine,
                                  KernelBuildIdentity currentBuild,
                                  KernelSignature signature) {
        return "matched".equals(compatibilityReason(currentMachine, currentBuild, signature));
    }

    /** Deterministic pretty JSON for reports, review, and diffing. */
    public String toJson() {
        try {
            return MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(toMap()) + "\n";
        } catch (IOException failure) {
            throw new IllegalStateException("Unable to serialize calibration profile", failure);
        }
    }

    public static KernelCalibrationProfile fromJson(String json) {
        if (json == null || json.isBlank()) {
            throw new IllegalArgumentException("Calibration profile JSON is empty");
        }
        try {
            JsonNode root = MAPPER.readTree(json);
            int schema = integer(root, "schemaVersion", -1);
            KernelMachineIdentity machine = machine(root.path("machine"));
            KernelBuildIdentity build = build(root.path("build"));
            List<KernelCalibrationEntry> entries = new ArrayList<>();
            JsonNode entryArray = root.path("entries");
            if (!entryArray.isArray()) {
                throw new IllegalArgumentException("entries must be an array");
            }
            for (JsonNode entry : entryArray) {
                List<KernelCalibrationCandidate> candidates = new ArrayList<>();
                JsonNode candidateArray = entry.path("candidates");
                if (candidateArray.isArray()) {
                    for (JsonNode candidate : candidateArray) {
                        candidates.add(candidate(candidate));
                    }
                }
                List<KernelTrustedBaseline> baselines = baselines(entry.path("trustedBaselines"));
                entries.add(new KernelCalibrationEntry(
                    text(entry, "kernelSha256", ""), text(entry, "kernelSignature", ""),
                    text(entry, "baselineVariant", ""), nullableText(entry, "winnerVariant"),
                    decimal(entry, "winnerSpeedup", 1.0),
                    text(entry, "decisionReason", ""), candidates, baselines));
            }
            return new KernelCalibrationProfile(schema, machine, build,
                text(root, "timestamp", ""), text(root, "methodologyVersion", ""), entries);
        } catch (IOException | RuntimeException failure) {
            throw new IllegalArgumentException("Invalid calibration profile", failure);
        }
    }

    private Map<String, Object> toMap() {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("schemaVersion", schemaVersion);
        root.put("timestamp", timestamp);
        root.put("methodologyVersion", methodologyVersion);
        root.put("machine", machineMap(machine));
        root.put("build", buildMap(build));
        root.put("entries", entries.stream().map(KernelCalibrationProfile::entryMap).toList());
        return root;
    }

    private static Map<String, Object> machineMap(KernelMachineIdentity machine) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (machine == null) return map;
        map.put("key", machine.key());
        map.put("architecture", machine.architecture());
        map.put("cpuVendor", machine.cpuVendor());
        map.put("cpuFamily", machine.cpuFamily());
        map.put("cpuModel", machine.cpuModel());
        map.put("cpuModelName", machine.cpuModelName());
        map.put("isaFeatures", machine.isaFeatures());
        map.put("os", machine.os());
        map.put("jvm", machine.jvm());
        map.put("nativeCompiler", machine.nativeCompiler());
        map.put("availableProcessors", machine.availableProcessors());
        return map;
    }

    private static Map<String, Object> buildMap(KernelBuildIdentity build) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (build == null) return map;
        map.put("key", build.key());
        map.put("jlcVersion", build.jlcVersion());
        map.put("gitSha", build.gitSha());
        map.put("kernelSignatureVersion", build.kernelSignatureVersion());
        map.put("variantSignatureVersion", build.variantSignatureVersion());
        map.put("codegenAbiVersion", build.codegenAbiVersion());
        map.put("compilerIdentity", build.compilerIdentity());
        map.put("strictFlags", build.strictFlags());
        return map;
    }

    private static Map<String, Object> entryMap(KernelCalibrationEntry entry) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("kernelSha256", entry.kernelSha256());
        map.put("kernelSignature", entry.kernelSignature());
        map.put("baselineVariant", entry.baselineVariant());
        map.put("winnerVariant", entry.winnerVariant());
        map.put("winnerSpeedup", entry.winnerSpeedup());
        map.put("decisionReason", entry.decisionReason());
        map.put("candidates", entry.candidates().stream().map(KernelCalibrationProfile::candidateMap).toList());
        map.put("trustedBaselines", entry.trustedBaselines().stream()
            .map(KernelCalibrationProfile::baselineMap).toList());
        return map;
    }

    private static Map<String, Object> baselineMap(KernelTrustedBaseline baseline) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("name", baseline.name());
        map.put("backend", baseline.backend());
        map.put("correctnessPassed", baseline.correctnessPassed());
        map.put("note", baseline.note());
        KernelBenchmarkStatistics stats = baseline.statistics();
        map.put("medianNanos", stats == null ? null : stats.medianNanos());
        map.put("minNanos", stats == null ? null : stats.minNanos());
        map.put("maxNanos", stats == null ? null : stats.maxNanos());
        map.put("madNanos", stats == null ? null : stats.madNanos());
        map.put("rawSamplesNanos", stats == null ? List.of() : stats.rawSamplesNanos());
        return map;
    }

    private static Map<String, Object> candidateMap(KernelCalibrationCandidate candidate) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("variantId", candidate.variantId());
        map.put("variantSignature", candidate.variantSignature());
        map.put("backend", candidate.backend());
        map.put("unroll", candidate.unroll());
        map.put("loopForm", candidate.loopForm());
        map.put("estimatedLiveVectorRegisters", candidate.estimatedLiveVectorRegisters());
        map.put("outcome", candidate.outcome());
        map.put("correctnessPassed", candidate.correctnessPassed());
        map.put("stable", candidate.stable());
        map.put("reason", candidate.reason());
        map.put("medianNanos", candidate.medianNanos());
        map.put("minNanos", candidate.minNanos());
        map.put("maxNanos", candidate.maxNanos());
        map.put("madNanos", candidate.madNanos());
        map.put("rawSamplesNanos", candidate.rawSamplesNanos());
        map.put("sourceGenerationNanos", candidate.sourceGenerationNanos());
        map.put("compilationNanos", candidate.compilationNanos());
        map.put("benchmarkNanos", candidate.benchmarkNanos());
        map.put("compiledTextBytes", candidate.compiledTextBytes());
        return map;
    }

    private static KernelCalibrationCandidate candidate(JsonNode node) {
        List<Long> samples = new ArrayList<>();
        JsonNode sampleArray = node.path("rawSamplesNanos");
        if (sampleArray.isArray()) {
            for (JsonNode sample : sampleArray) samples.add(sample.asLong());
        }
        return new KernelCalibrationCandidate(
            text(node, "variantId", ""), text(node, "variantSignature", ""),
            text(node, "backend", ""), integer(node, "unroll", 1),
            text(node, "loopForm", ""), integer(node, "estimatedLiveVectorRegisters", 0),
            text(node, "outcome", ""), node.path("correctnessPassed").asBoolean(false),
            node.path("stable").asBoolean(false), text(node, "reason", ""),
            nullableDecimal(node, "medianNanos"), nullableDecimal(node, "minNanos"),
            nullableDecimal(node, "maxNanos"), nullableDecimal(node, "madNanos"), samples,
            longValue(node, "sourceGenerationNanos", 0L), longValue(node, "compilationNanos", 0L),
            longValue(node, "benchmarkNanos", 0L), longValue(node, "compiledTextBytes", 0L));
    }

    private static List<KernelTrustedBaseline> baselines(JsonNode nodes) {
        if (nodes == null || !nodes.isArray()) return List.of();
        List<KernelTrustedBaseline> result = new ArrayList<>();
        for (JsonNode node : nodes) {
            List<Long> samples = new ArrayList<>();
            JsonNode sampleArray = node.path("rawSamplesNanos");
            if (sampleArray.isArray()) {
                for (JsonNode sample : sampleArray) samples.add(sample.asLong());
            }
            KernelBenchmarkStatistics stats = samples.isEmpty()
                ? null : KernelBenchmarkStatistics.from(samples);
            result.add(new KernelTrustedBaseline(text(node, "name", "unknown"),
                text(node, "backend", "unknown"),
                node.path("correctnessPassed").asBoolean(false), stats,
                text(node, "note", "")));
        }
        return result;
    }

    private static KernelMachineIdentity machine(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isEmpty()) return null;
        return new KernelMachineIdentity(text(node, "architecture", "unknown"),
            text(node, "cpuVendor", "unknown"), text(node, "cpuFamily", "unknown"),
            text(node, "cpuModel", "unknown"), text(node, "cpuModelName", "unknown"),
            text(node, "isaFeatures", "unknown"), text(node, "os", "unknown"),
            text(node, "jvm", "unknown"), text(node, "nativeCompiler", "unknown"),
            integer(node, "availableProcessors", 0));
    }

    private static KernelBuildIdentity build(JsonNode node) {
        if (node == null || node.isMissingNode() || node.isEmpty()) return null;
        return new KernelBuildIdentity(text(node, "jlcVersion", "unknown"),
            text(node, "gitSha", "unknown"), text(node, "kernelSignatureVersion", "unknown"),
            text(node, "variantSignatureVersion", "unknown"),
            text(node, "compilerIdentity", "unknown"),
            text(node, "codegenAbiVersion", KernelBuildIdentity.CODEGEN_ABI_VERSION),
            text(node, "strictFlags", KernelBuildIdentity.STRICT_FLAGS));
    }

    private static String text(JsonNode node, String field, String fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() ? fallback : value.asText(fallback);
    }

    private static String nullableText(JsonNode node, String field) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }

    private static int integer(JsonNode node, String field, int fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || !value.isNumber() ? fallback : value.asInt(fallback);
    }

    private static long longValue(JsonNode node, String field, long fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || !value.isNumber() ? fallback : value.asLong(fallback);
    }

    private static double decimal(JsonNode node, String field, double fallback) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || !value.isNumber() ? fallback : value.asDouble(fallback);
    }

    private static Double nullableDecimal(JsonNode node, String field) {
        JsonNode value = node == null ? null : node.get(field);
        return value == null || value.isNull() || !value.isNumber() ? null : value.asDouble();
    }
}
