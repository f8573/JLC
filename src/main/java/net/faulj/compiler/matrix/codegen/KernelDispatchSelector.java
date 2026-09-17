package net.faulj.compiler.matrix.codegen;

import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The single runtime authority for resolving a semantic kernel to a backend.
 * Profile I/O, compatibility checks, ISA probing, and exact artifact lookup
 * occur at selector construction/resolution; repeated execution uses a cached
 * immutable {@link Selection}.
 */
public final class KernelDispatchSelector {
    private static final AtomicReference<KernelDispatchSelector> GLOBAL = new AtomicReference<>();

    private final RuntimeEnvironment environment;
    private final Path profilePath;
    private final Optional<KernelCalibrationProfile> profile;
    private final String profileLoadReason;
    private final ConcurrentHashMap<KernelSignature, Selection> resolved = new ConcurrentHashMap<>();

    /** Production constructor that loads and validates an explicit profile path. */
    public KernelDispatchSelector(GeneratedKernelRegistry registry,
                                  KernelMachineIdentity machine,
                                  KernelBuildIdentity build,
                                  Path profilePath) {
        this(RuntimeEnvironment.current(registry, machine, build), profilePath);
    }

    public KernelDispatchSelector(RuntimeEnvironment environment,
                                  Path profilePath) {
        this.environment = environment == null ? RuntimeEnvironment.current() : environment;
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

    /** Constructor for tests and embedding code with an already parsed profile. */
    public KernelDispatchSelector(RuntimeEnvironment environment,
                                  KernelCalibrationProfile profile) {
        this.environment = environment == null ? RuntimeEnvironment.current() : environment;
        this.profilePath = null;
        this.profile = Optional.ofNullable(profile);
        this.profileLoadReason = profile == null ? "no profile configured" : "provided profile";
    }

    public static KernelDispatchSelector global() {
        String configured = System.getProperty("jlc.compiler.autotune.profile");
        String normalized = configured == null ? "" : configured.trim();
        KernelDispatchSelector current = GLOBAL.get();
        if (current != null && current.profilePathText().equals(normalized)) {
            return current;
        }
        KernelDispatchSelector next = new KernelDispatchSelector(
            RuntimeEnvironment.current(GeneratedKernelRegistry.global(),
                KernelMachineIdentity.current(), KernelBuildIdentity.current()),
            normalized.isBlank() ? null : Path.of(normalized));
        GLOBAL.set(next);
        return next;
    }

    public static void resetForTests() {
        GLOBAL.set(null);
    }

    /** Compatibility view for existing R5 diagnostics. */
    public Selection select(KernelSignature signature) {
        if (signature == null) return Selection.none("kernel signature is null");
        return resolved.computeIfAbsent(signature, this::resolve);
    }

    /** Typed production API; Java is represented by a real choice subtype. */
    public BackendChoice selectChoice(KernelSignature signature) {
        return select(signature).choice();
    }

    /** Alias used by callers that prefer the backend terminology. */
    public BackendChoice selectBackend(KernelSignature signature) {
        return selectChoice(signature);
    }

    /** Explicit, dependency-injected selection entry point for integration tests. */
    public BackendChoice select(KernelSignature signature,
                                RuntimeEnvironment runtime,
                                KernelCalibrationProfile calibration) {
        return new KernelDispatchSelector(runtime, calibration).selectChoice(signature);
    }

    public String explain(KernelSignature signature) {
        return select(signature).explain();
    }

    public String profileLoadReason() {
        return profileLoadReason;
    }

    public RuntimeEnvironment environment() {
        return environment;
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
        if (profile.isPresent()) {
            KernelCalibrationProfile calibration = profile.get();
            String compatibility = calibration.compatibilityReason(
                environment.machine(), environment.build(), signature);
            if ("matched".equals(compatibility)) {
                KernelCalibrationEntry entry = calibration.entry(signature).orElse(null);
                if (entry != null
                    && entry.selectionStatus() == BackendSelectionStatus.NO_STABLE_WINNER) {
                    return coldStart(signature,
                        "profile recorded no stable winner; using fallback policy");
                }
                if (entry != null && entry.winnerChoice() != null) {
                    BackendCandidateEvidence evidence = evidence(entry, entry.winnerChoice());
                    if (evidence != null && evidence.valid()
                        && environment.canExecute(signature, entry.winnerChoice())) {
                        GeneratedKernelRegistry.Entry registered = registered(entry.winnerChoice());
                        return new Selection(entry.winnerChoice(), registered, "profile",
                            "profile hit", true, true, evidence.stable(),
                            median(evidence), entry.winnerSpeedup());
                    }
                    String reason = evidence == null
                        ? "profile winner has no retained evidence"
                        : "profile winner unavailable: "
                            + environment.validationReason(signature, entry.winnerChoice());
                    return coldStart(signature, reason);
                }
                return coldStart(signature, "profile entry has no typed winner");
            }
            return coldStart(signature, "profile ignored: " + compatibility);
        }
        return coldStart(signature, profileLoadReason);
    }

    private Selection coldStart(KernelSignature signature, String reason) {
        BackendChoice baseline = new GeneratedAvx2Backend(
            KernelVariantSignature.baselineAvx2(signature));
        if (environment.canExecute(signature, baseline)) {
            return selected(baseline, "cold-start",
                reason + "; selected baseline AVX2", false, false, true,
                Double.NaN, 1.0);
        }

        BackendChoice scalar = scalarFallback(signature);
        if (scalar != null && environment.canExecute(signature, scalar)) {
            return selected(scalar, "cold-start",
                reason + "; selected scalar-native fallback", false, false, true,
                Double.NaN, 1.0);
        }

        BackendChoice java = new R2JavaBackend();
        if (environment.canExecute(signature, java)) {
            return selected(java, "fallback",
                reason + "; selected R2 Java fallback", false, false, true,
                Double.NaN, 1.0);
        }
        return Selection.none(reason + "; no executable backend");
    }

    private BackendChoice scalarFallback(KernelSignature signature) {
        KernelVariantSignature legacy = KernelVariantSignature.legacyScalar(signature);
        if (environment.registry().lookup(legacy) != null) {
            return new ScalarNativeBackend(legacy);
        }
        return environment.registry().entriesFor(signature).stream()
            .filter(entry -> entry.descriptor().backend() == CodegenBackend.SCALAR_CPP)
            .map(entry -> (BackendChoice) new ScalarNativeBackend(
                entry.descriptor().variantSignature()))
            .filter(choice -> environment.canExecute(signature, choice))
            .findFirst().orElse(null);
    }

    private Selection selected(BackendChoice choice,
                               String source,
                               String reason,
                               boolean profileHit,
                               boolean machineMatched,
                               boolean stable,
                               double medianNanos,
                               double speedup) {
        return new Selection(choice, registered(choice), source, reason, profileHit,
            machineMatched, stable, medianNanos, speedup);
    }

    private GeneratedKernelRegistry.Entry registered(BackendChoice choice) {
        return choice == null ? null : choice.variantOptional()
            .map(environment.registry()::lookup).orElse(null);
    }

    private static BackendCandidateEvidence evidence(KernelCalibrationEntry entry,
                                                     BackendChoice choice) {
        BackendCandidateEvidence direct = entry.evidence(choice);
        if (direct != null) return direct;
        String variantText = choice.variantOptional()
            .map(KernelVariantSignature::canonicalText).orElse(null);
        if (variantText == null) return null;
        for (KernelCalibrationCandidate candidate : entry.candidates()) {
            if (variantText.equals(candidate.variantSignature())) {
                KernelBenchmarkStatistics stats = candidate.rawSamplesNanos().isEmpty()
                    ? null : KernelBenchmarkStatistics.from(candidate.rawSamplesNanos());
                return new BackendCandidateEvidence(choice,
                    outcome(candidate.outcome()), candidate.correctnessPassed(), candidate.stable(),
                    stats, candidate.benchmarkNanos(), candidate.reason());
            }
        }
        return null;
    }

    private static KernelCandidateOutcome outcome(String value) {
        try {
            return KernelCandidateOutcome.valueOf(value);
        } catch (RuntimeException ignored) {
            return KernelCandidateOutcome.BENCHMARK_FAILED;
        }
    }

    private static double median(BackendCandidateEvidence evidence) {
        return evidence.statistics() == null ? Double.NaN : evidence.statistics().medianNanos();
    }

    private String profilePathText() {
        return profilePath == null ? "" : profilePath.toString();
    }

    /** Immutable result retained as a compatibility diagnostic around the typed choice. */
    public static final class Selection {
        private final BackendChoice choice;
        private final GeneratedKernelRegistry.Entry entry;
        private final String source;
        private final String reason;
        private final boolean profileHit;
        private final boolean machineMatched;
        private final boolean stable;
        private final double medianNanos;
        private final double speedupOverBaseline;

        private Selection(BackendChoice choice,
                          GeneratedKernelRegistry.Entry entry,
                          String source,
                          String reason,
                          boolean profileHit,
                          boolean machineMatched,
                          boolean stable,
                          double medianNanos,
                          double speedupOverBaseline) {
            this.choice = choice;
            this.entry = entry;
            this.source = source;
            this.reason = reason;
            this.profileHit = profileHit;
            this.machineMatched = machineMatched;
            this.stable = stable;
            this.medianNanos = medianNanos;
            this.speedupOverBaseline = speedupOverBaseline;
        }

        /** Compatibility constructor for the generated-only R5 result shape. */
        @Deprecated
        public Selection(KernelVariantSignature variant,
                          GeneratedKernelRegistry.Entry entry,
                          String source,
                          String reason,
                          boolean profileHit,
                          boolean machineMatched,
                          boolean stable,
                          double medianNanos,
                          double speedupOverBaseline) {
            this(legacyChoice(variant), entry, source, reason, profileHit,
                machineMatched, stable, medianNanos, speedupOverBaseline);
        }

        public static Selection none(String reason) {
            return new Selection((BackendChoice) null, null, "fallback", reason, false, false,
                false, Double.NaN, 1.0);
        }

        public BackendChoice choice() { return choice; }

        /** Deprecated generated-only view; Java choices intentionally return null here. */
        @Deprecated
        public KernelVariantSignature variant() {
            return choice == null ? null : choice.variantOptional().orElse(null);
        }

        public GeneratedKernelRegistry.Entry entry() { return entry; }
        public String source() { return source; }
        public String reason() { return reason; }
        public boolean profileHit() { return profileHit; }
        public boolean machineMatched() { return machineMatched; }
        public boolean stable() { return stable; }
        public double medianNanos() { return medianNanos; }
        public double speedupOverBaseline() { return speedupOverBaseline; }

        public boolean available() {
            return choice != null || entry != null;
        }

        public String explain() {
            String variant = choice == null ? "<none>" : choice.variantOptional()
                .map(KernelVariantSignature::variantId).orElse("<java>");
            String kernel = choice != null && choice.variantOptional().isPresent()
                ? choice.variantOptional().get().kernelSignature().shortHash() : "<java/fallback>";
            return "kernel dispatch:\n"
                + "  kernel=" + kernel + '\n'
                + "  profile=" + (profileHit ? "hit" : "miss") + '\n'
                + "  machineMatched=" + machineMatched + '\n'
                + "  source=" + source + '\n'
                + "  choice=" + (choice == null ? "<fallback>" : choice.kind().tag()) + '\n'
                + "  selected=" + variant + '\n'
                + "  medianNanos=" + medianNanos + '\n'
                + "  measuredSpeedup=" + speedupOverBaseline + '\n'
                + "  reason=" + reason;
        }

        private static BackendChoice legacyChoice(KernelVariantSignature variant) {
            if (variant == null) return null;
            return variant.backend() == CodegenBackend.AVX2
                ? new GeneratedAvx2Backend(variant) : new ScalarNativeBackend(variant);
        }
    }
}
