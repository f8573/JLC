package net.faulj.compiler.matrix.codegen;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Profile-aware, no-benchmark runtime selector for registered generated
 * variants. Profile parsing happens once; selection results are cached by
 * exact semantic KernelSignature.
 */
public final class KernelDispatchSelector {
    private static final AtomicReference<KernelDispatchSelector> GLOBAL = new AtomicReference<>();

    private final GeneratedKernelRegistry registry;
    private final KernelMachineIdentity machine;
    private final KernelBuildIdentity build;
    private final Path profilePath;
    private final Optional<KernelCalibrationProfile> profile;
    private final String profileLoadReason;
    private final ConcurrentHashMap<KernelSignature, Selection> resolved = new ConcurrentHashMap<>();

    public KernelDispatchSelector(GeneratedKernelRegistry registry,
                                  KernelMachineIdentity machine,
                                  KernelBuildIdentity build,
                                  Path profilePath) {
        this.registry = registry == null ? GeneratedKernelRegistry.global() : registry;
        this.machine = machine == null ? KernelMachineIdentity.current() : machine;
        this.build = build == null ? KernelBuildIdentity.current() : build;
        this.profilePath = profilePath;
        if (profilePath == null) {
            this.profile = Optional.empty();
            this.profileLoadReason = "no profile configured";
        } else {
            Optional<KernelCalibrationProfile> loaded = KernelCalibrationProfileStore.load(profilePath);
            this.profile = loaded;
            this.profileLoadReason = loaded.isPresent() ? "loaded" : "missing or corrupt profile";
        }
    }

    public static KernelDispatchSelector global() {
        String configured = System.getProperty("jlc.compiler.autotune.profile");
        String normalized = configured == null ? "" : configured.trim();
        KernelDispatchSelector current = GLOBAL.get();
        if (current != null && current.profilePathText().equals(normalized)) {
            return current;
        }
        KernelDispatchSelector next = new KernelDispatchSelector(
            GeneratedKernelRegistry.global(), KernelMachineIdentity.current(),
            KernelBuildIdentity.current(), normalized.isBlank() ? null : Path.of(normalized));
        GLOBAL.set(next);
        return next;
    }

    public static void resetForTests() {
        GLOBAL.set(null);
    }

    public Selection select(KernelSignature signature) {
        if (signature == null) {
            return Selection.none("kernel signature is null");
        }
        return resolved.computeIfAbsent(signature, this::resolve);
    }

    public String explain(KernelSignature signature) {
        return select(signature).explain();
    }

    public String profileLoadReason() {
        return profileLoadReason;
    }

    public int resolvedCacheSize() {
        return resolved.size();
    }

    /** Small diagnostic for the cached lookup path, not a tuning measurement. */
    public long measureLookupNanos(KernelSignature signature, int repetitions) {
        if (repetitions < 1) throw new IllegalArgumentException("repetitions must be positive");
        select(signature);
        long start = System.nanoTime();
        for (int index = 0; index < repetitions; index++) select(signature);
        return Math.max(1L, System.nanoTime() - start) / repetitions;
    }

    private Selection resolve(KernelSignature signature) {
        List<GeneratedKernelRegistry.Entry> entries = registry.entriesFor(signature);
        if (entries.isEmpty()) {
            return Selection.none("no registered generated variants");
        }

        if (profile.isPresent()) {
            KernelCalibrationProfile calibration = profile.get();
            String compatibility = calibration.compatibilityReason(machine, build, signature);
            if ("matched".equals(compatibility)) {
                KernelCalibrationEntry entry = calibration.entry(signature).orElse(null);
                if (entry != null && entry.winnerVariant() != null) {
                    KernelCalibrationCandidate evidence = entry.candidates().stream()
                        .filter(candidate -> entry.winnerVariant().equals(candidate.variantSignature()))
                        .findFirst().orElse(null);
                    GeneratedKernelRegistry.Entry registered = entries.stream()
                        .filter(item -> entry.winnerVariant().equals(
                            item.descriptor().variantSignature().canonicalText()))
                        .filter(item -> isAvailable(item.descriptor()))
                        .findFirst().orElse(null);
                    if (registered != null && evidence != null
                        && evidence.correctnessPassed() && evidence.stable()) {
                        double median = evidence.medianNanos() == null
                            ? Double.NaN : evidence.medianNanos();
                        return new Selection(registered.descriptor().variantSignature(), registered,
                            "profile", "profile hit", true, true,
                            evidence.stable(), median, entry.winnerSpeedup());
                    }
                    return coldStart(signature, entries,
                        "profile winner unavailable, unstable, or not correctness-gated");
                }
                return coldStart(signature, entries, "profile entry has no winner");
            }
            return coldStart(signature, entries, "profile ignored: " + compatibility);
        }
        return coldStart(signature, entries, profileLoadReason);
    }

    private Selection coldStart(KernelSignature signature,
                                List<GeneratedKernelRegistry.Entry> entries,
                                String reason) {
        GeneratedKernelRegistry.Entry avx2 = registry.lookup(
            KernelVariantSignature.baselineAvx2(signature));
        if (avx2 != null && isAvailable(avx2.descriptor())) {
            return new Selection(avx2.descriptor().variantSignature(), avx2,
                "cold-start", reason + "; selected R4 BASELINE_AVX2", false, true,
                true, Double.NaN, 1.0);
        }
        GeneratedKernelRegistry.Entry scalar = registry.lookup(
            KernelVariantSignature.legacyScalar(signature));
        if (scalar == null) {
            scalar = registry.lookup(signature, CodegenBackend.SCALAR_CPP);
        }
        if (scalar != null && isAvailable(scalar.descriptor())) {
            return new Selection(scalar.descriptor().variantSignature(), scalar,
                "cold-start", reason + "; selected scalar fallback", false, true,
                true, Double.NaN, 1.0);
        }
        return Selection.none(reason + "; no available baseline variant");
    }

    private static boolean isAvailable(GeneratedKernelDescriptor descriptor) {
        if (descriptor.backend() == CodegenBackend.AVX2) {
            if (!RuntimeCpuFeatures.avx2Supported()) return false;
            return !descriptor.requiredCpuFeature().contains("fma")
                || RuntimeCpuFeatures.fmaSupported();
        }
        return true;
    }

    private String profilePathText() {
        return profilePath == null ? "" : profilePath.toString();
    }

    public record Selection(KernelVariantSignature variant,
                            GeneratedKernelRegistry.Entry entry,
                            String source,
                            String reason,
                            boolean profileHit,
                            boolean machineMatched,
                            boolean stable,
                            double medianNanos,
                            double speedupOverBaseline) {
        public static Selection none(String reason) {
            return new Selection(null, null, "fallback", reason, false, false,
                false, Double.NaN, 1.0);
        }

        public boolean available() {
            return entry != null;
        }

        public String explain() {
            return "kernel dispatch:\n"
                + "  kernel=" + (variant == null ? "<none>" : variant.kernelSignature().shortHash()) + '\n'
                + "  profile=" + (profileHit ? "hit" : "miss") + '\n'
                + "  machineMatched=" + machineMatched + '\n'
                + "  source=" + source + '\n'
                + "  selected=" + (variant == null ? "<fallback>" : variant.variantId()) + '\n'
                + "  medianNanos=" + medianNanos + '\n'
                + "  measuredSpeedup=" + speedupOverBaseline + '\n'
                + "  reason=" + reason;
        }
    }
}
