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
 * Human-inspectable machine/build-specific backend calibration profile.
 *
 * <p>The R5 schema remains visible for review and compatibility, while the
 * dispatch schema adds a typed {@code KernelSignature -> BackendChoice} record
 * and comparable evidence for Java, scalar-native, and generated AVX2 paths.
 * Old R5-only JSON is deliberately rejected by compatibility checks.</p>
 */
public final class KernelCalibrationProfile {
    public static final int CURRENT_SCHEMA_VERSION = 1;
    /** Version 3 makes the explicit calibrated/no-stable winner invariant part of the wire contract. */
    public static final int CURRENT_DISPATCH_SCHEMA_VERSION = 3;
    public static final String METHODOLOGY_VERSION = "r5-median-mad-v1";

    private static final ObjectMapper MAPPER = new ObjectMapper()
        .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    private final int schemaVersion;
    private final int dispatchSchemaVersion;
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
        this(CURRENT_SCHEMA_VERSION, CURRENT_DISPATCH_SCHEMA_VERSION, machine, build,
            timestamp, methodologyVersion, entries);
    }

    private KernelCalibrationProfile(int schemaVersion,
                                     int dispatchSchemaVersion,
                                     KernelMachineIdentity machine,
                                     KernelBuildIdentity build,
                                     String timestamp,
                                     String methodologyVersion,
                                     List<KernelCalibrationEntry> entries) {
        this.schemaVersion = schemaVersion;
        this.dispatchSchemaVersion = dispatchSchemaVersion;
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

    public static KernelCalibrationProfile fromBackendResult(KernelMachineIdentity machine,
                                                             KernelBuildIdentity build,
                                                             BackendCalibrationResult result) {
        if (machine == null || build == null || result == null) {
            throw new IllegalArgumentException("Machine, build, and backend result are required");
        }
        return new KernelCalibrationProfile(machine, build,
            java.time.Instant.now().toString(), METHODOLOGY_VERSION,
            List.of(KernelCalibrationEntry.from(result)));
    }

    /** Replace one exact signature/bucket while retaining every unrelated entry. */
    public KernelCalibrationProfile merge(BackendCalibrationResult result) {
        if (result == null) throw new IllegalArgumentException("Backend result is required");
        KernelCalibrationEntry replacement = KernelCalibrationEntry.from(result);
        List<KernelCalibrationEntry> merged = new ArrayList<>();
        boolean replaced = false;
        for (KernelCalibrationEntry entry : entries) {
            if (entry.kernelSha256().equals(replacement.kernelSha256())
                && entry.kernelSignature().equals(replacement.kernelSignature())
                && entry.workloadBucket().equals(replacement.workloadBucket())) {
                if (!replaced) {
                    merged.add(replacement);
                    replaced = true;
                }
            } else {
                merged.add(entry);
            }
        }
        if (!replaced) merged.add(replacement);
        return new KernelCalibrationProfile(CURRENT_SCHEMA_VERSION,
            CURRENT_DISPATCH_SCHEMA_VERSION, machine, build,
            java.time.Instant.now().toString(), methodologyVersion, merged);
    }

    /** Alias emphasizing that this operation is an update, not a new profile. */
    public KernelCalibrationProfile withBackendResult(BackendCalibrationResult result) {
        return merge(result);
    }

    public int schemaVersion() { return schemaVersion; }
    public int dispatchSchemaVersion() { return dispatchSchemaVersion; }
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

    public Optional<KernelCalibrationEntry> entry(KernelSignature signature,
                                                  String workloadBucket) {
        if (signature == null) return Optional.empty();
        String bucket = workloadBucket == null ? "exact" : workloadBucket;
        return entries.stream().filter(entry -> signature.sha256().equals(entry.kernelSha256())
            && signature.canonicalText().equals(entry.kernelSignature())
            && bucket.equals(entry.workloadBucket())).findFirst();
    }

    public String compatibilityReason(KernelMachineIdentity currentMachine,
                                      KernelBuildIdentity currentBuild,
                                      KernelSignature signature) {
        if (schemaVersion != CURRENT_SCHEMA_VERSION) return "schema version mismatch";
        if (dispatchSchemaVersion != CURRENT_DISPATCH_SCHEMA_VERSION) {
            return "dispatch schema version mismatch";
        }
        if (!METHODOLOGY_VERSION.equals(methodologyVersion)) {
            return "methodology version mismatch";
        }
        if (build == null || !build.identityVerified() || build.gitDirty()
            || currentBuild == null || !currentBuild.identityVerified() || currentBuild.gitDirty()) {
            return "build identity unverified";
        }
        if (machine == null || currentMachine == null || !machine.key().equals(currentMachine.key())) {
            return "machine mismatch";
        }
        if (!build.key().equals(currentBuild.key())) {
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
            int dispatchSchema = integer(root, "dispatchSchemaVersion", -1);
            KernelMachineIdentity machine = machine(root.path("machine"));
            KernelBuildIdentity build = build(root.path("build"));
            List<KernelCalibrationEntry> entries = new ArrayList<>();
            JsonNode entryArray = root.path("entries");
            if (!entryArray.isArray()) {
                throw new IllegalArgumentException("entries must be an array");
            }
            for (JsonNode entry : entryArray) {
                String kernelSha = text(entry, "kernelSha256", "");
                String kernelText = text(entry, "kernelSignature", "");
                KernelSignature signature = KernelSignature.fromCanonicalText(kernelText);
                if (!signature.sha256().equals(kernelSha)) {
                    throw new IllegalArgumentException("Kernel signature digest mismatch");
                }
                List<KernelCalibrationCandidate> candidates = new ArrayList<>();
                JsonNode candidateArray = entry.path("candidates");
                if (candidateArray.isArray()) {
                    for (JsonNode candidate : candidateArray) {
                        candidates.add(candidate(candidate));
                    }
                }
                List<KernelTrustedBaseline> baselines = baselines(entry.path("trustedBaselines"));
                BackendChoice baselineChoice = choice(entry.get("baselineChoice"), signature);
                BackendChoice winnerChoice = choice(entry.get("winnerChoice"), signature);
                BackendSelectionStatus selectionStatus = selectionStatus(entry, winnerChoice);
                if (!entry.has("baselineChoice")) {
                    baselineChoice = legacyChoice(signature, text(entry, "baselineVariant", ""));
                }
                if (!entry.has("winnerChoice")
                    && selectionStatus != BackendSelectionStatus.NO_STABLE_WINNER) {
                    winnerChoice = legacyChoice(signature, nullableText(entry, "winnerVariant"));
                }
                List<BackendCandidateEvidence> backendCandidates = backendCandidates(
                    entry.path("backendCandidates"), signature);
                double winnerSpeedup = decimal(entry, "winnerSpeedup",
                    selectionStatus == BackendSelectionStatus.NO_STABLE_WINNER ? 0.0 : 1.0);
                boolean stable = entry.has("stable")
                    ? entry.path("stable").asBoolean(false)
                    : selectionStatus == BackendSelectionStatus.CALIBRATED;
                entries.add(KernelCalibrationEntry.persisted(
                    kernelSha, kernelText, text(entry, "workloadBucket", "exact"),
                    text(entry, "baselineVariant", ""), nullableText(entry, "winnerVariant"),
                    winnerSpeedup, text(entry, "decisionReason", ""), candidates, baselines,
                    baselineChoice, winnerChoice, backendCandidates, selectionStatus, stable));
            }
            return new KernelCalibrationProfile(schema, dispatchSchema, machine, build,
                text(root, "timestamp", ""), text(root, "methodologyVersion", ""), entries);
        } catch (IOException | RuntimeException failure) {
            throw new IllegalArgumentException("Invalid calibration profile", failure);
        }
    }

    private Map<String, Object> toMap() {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("schemaVersion", schemaVersion);
        root.put("dispatchSchemaVersion", dispatchSchemaVersion);
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
        map.put("gitDirty", build.gitDirty());
        map.put("identityVerified", build.identityVerified());
        map.put("javaCompiler", build.javaCompiler());
        map.put("javaRuntime", build.javaRuntime());
        map.put("nativeCompilerVersion", build.nativeCompilerVersion());
        map.put("nativeVendor", build.nativeVendor());
        return map;
    }

    private static Map<String, Object> entryMap(KernelCalibrationEntry entry) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("kernelSha256", entry.kernelSha256());
        map.put("kernelSignature", entry.kernelSignature());
        map.put("workloadBucket", entry.workloadBucket());
        map.put("baselineChoice", entry.baselineChoice() == null
            ? null : BackendChoiceCodec.toMap(entry.baselineChoice()));
        map.put("winnerChoice", entry.winnerChoice() == null
            ? null : BackendChoiceCodec.toMap(entry.winnerChoice()));
        map.put("selectionStatus", entry.selectionStatus().name());
        map.put("stable", entry.stable());
        map.put("baselineVariant", entry.baselineVariant());
        map.put("winnerVariant", entry.winnerVariant());
        map.put("winnerSpeedup", entry.winnerSpeedup());
        map.put("decisionReason", entry.decisionReason());
        map.put("backendCandidates", entry.backendCandidates().stream()
            .map(KernelCalibrationProfile::backendCandidateMap).toList());
        map.put("candidates", entry.candidates().stream()
            .map(KernelCalibrationProfile::candidateMap).toList());
        map.put("trustedBaselines", entry.trustedBaselines().stream()
            .map(KernelCalibrationProfile::baselineMap).toList());
        return map;
    }

    private static Map<String, Object> backendCandidateMap(BackendCandidateEvidence candidate) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("choice", BackendChoiceCodec.toMap(candidate.choice()));
        map.put("outcome", candidate.outcome().name());
        map.put("correctnessPassed", candidate.correctnessPassed());
        map.put("stable", candidate.stable());
        map.put("reason", candidate.reason());
        map.put("benchmarkNanos", candidate.benchmarkNanos());
        KernelBenchmarkStatistics stats = candidate.statistics();
        map.put("medianNanos", stats == null ? null : stats.medianNanos());
        map.put("minNanos", stats == null ? null : stats.minNanos());
        map.put("maxNanos", stats == null ? null : stats.maxNanos());
        map.put("madNanos", stats == null ? null : stats.madNanos());
        map.put("rawSamplesNanos", stats == null ? List.of() : stats.rawSamplesNanos());
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
        List<Long> samples = samples(node.path("rawSamplesNanos"));
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

    private static List<BackendCandidateEvidence> backendCandidates(JsonNode nodes,
                                                                     KernelSignature signature) {
        if (nodes == null || !nodes.isArray()) return List.of();
        List<BackendCandidateEvidence> result = new ArrayList<>();
        for (JsonNode node : nodes) {
            BackendChoice choice = BackendChoiceCodec.fromNode(node.path("choice"), signature);
            KernelCandidateOutcome outcome;
            try {
                outcome = KernelCandidateOutcome.valueOf(text(node, "outcome", ""));
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException("Invalid backend candidate outcome", failure);
            }
            List<Long> raw = samples(node.path("rawSamplesNanos"));
            KernelBenchmarkStatistics stats = raw.isEmpty()
                ? null : KernelBenchmarkStatistics.from(raw);
            result.add(new BackendCandidateEvidence(choice, outcome,
                node.path("correctnessPassed").asBoolean(false),
                node.path("stable").asBoolean(false), stats,
                longValue(node, "benchmarkNanos", 0L), text(node, "reason", "")));
        }
        return List.copyOf(result);
    }

    private static BackendChoice choice(JsonNode node, KernelSignature signature) {
        if (node == null || node.isNull() || node.isMissingNode()) return null;
        return BackendChoiceCodec.fromNode(node, signature);
    }

    private static BackendSelectionStatus selectionStatus(JsonNode node,
                                                           BackendChoice winnerChoice) {
        JsonNode value = node == null ? null : node.get("selectionStatus");
        if (value == null || value.isNull() || value.asText().isBlank()) {
            return winnerChoice == null ? BackendSelectionStatus.NO_STABLE_WINNER
                : BackendSelectionStatus.CALIBRATED;
        }
        try {
            return BackendSelectionStatus.valueOf(value.asText());
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("Invalid backend selection status", failure);
        }
    }

    private static BackendChoice legacyChoice(KernelSignature signature, String variantText) {
        if (variantText == null || variantText.isBlank()) return null;
        KernelVariantSignature variant = KernelVariantSignature.fromCanonicalText(signature, variantText);
        return variant.backend() == CodegenBackend.AVX2
            ? new GeneratedAvx2Backend(variant) : new ScalarNativeBackend(variant);
    }

    private static List<KernelTrustedBaseline> baselines(JsonNode nodes) {
        if (nodes == null || !nodes.isArray()) return List.of();
        List<KernelTrustedBaseline> result = new ArrayList<>();
        for (JsonNode node : nodes) {
            List<Long> samples = samples(node.path("rawSamplesNanos"));
            KernelBenchmarkStatistics stats = samples.isEmpty()
                ? null : KernelBenchmarkStatistics.from(samples);
            result.add(new KernelTrustedBaseline(text(node, "name", "unknown"),
                text(node, "backend", "unknown"),
                node.path("correctnessPassed").asBoolean(false), stats,
                text(node, "note", "")));
        }
        return result;
    }

    private static List<Long> samples(JsonNode sampleArray) {
        if (sampleArray == null || !sampleArray.isArray()) return List.of();
        List<Long> samples = new ArrayList<>();
        for (JsonNode sample : sampleArray) samples.add(sample.asLong());
        return samples;
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
            text(node, "strictFlags", KernelBuildIdentity.STRICT_FLAGS),
            node.path("gitDirty").asBoolean(false),
            node.path("identityVerified").asBoolean(false),
            text(node, "javaCompiler", "unknown"), text(node, "javaRuntime", "unknown"),
            text(node, "nativeCompilerVersion", "unknown"),
            text(node, "nativeVendor", "unknown"));
    }

    private static KernelCalibrationEntry persisted(String kernelSha256,
                                                    String kernelSignature,
                                                    String workloadBucket,
                                                    String baselineVariant,
                                                    String winnerVariant,
                                                    double winnerSpeedup,
                                                    String decisionReason,
                                                    List<KernelCalibrationCandidate> candidates,
                                                    List<KernelTrustedBaseline> trustedBaselines,
                                                    BackendChoice baselineChoice,
                                                    BackendChoice winnerChoice,
                                                    List<BackendCandidateEvidence> backendCandidates,
                                                    BackendSelectionStatus selectionStatus,
                                                    boolean stable) {
        return KernelCalibrationEntry.persisted(kernelSha256, kernelSignature, workloadBucket,
            baselineVariant, winnerVariant, winnerSpeedup, decisionReason, candidates,
            trustedBaselines, baselineChoice, winnerChoice, backendCandidates,
            selectionStatus, stable);
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
