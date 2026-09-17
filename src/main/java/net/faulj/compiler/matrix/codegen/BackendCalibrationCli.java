package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.concurrent.TimeUnit;

import net.faulj.compiler.matrix.CompiledMatrixProgram;
import net.faulj.compiler.matrix.FlopCostModel;
import net.faulj.compiler.matrix.MatrixCompiler;
import net.faulj.compiler.matrix.MatrixExpr;
import net.faulj.compiler.matrix.OptimizationSemantics;
import net.faulj.compiler.matrix.cpu.CpuFusedRegionStep;
import net.faulj.compiler.matrix.cpu.FusionStrategy;
import net.faulj.compiler.matrix.affine.LogicalBuffer;
import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelLowerer;
import net.faulj.compiler.matrix.kernel.KernelLoweringResult;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeGeneratedKernelSupport;

/**
 * Standalone, explicit backend calibration entry point.
 *
 * <p>Each generated source is built into a temporary JNI library and then
 * invoked through the same whole-region boundary used by production dispatch.
 * The temporary build is removed after the profile update; only the profile
 * and its atomic replacement remain.</p>
 */
public final class BackendCalibrationCli {
    private static volatile long sink;

    private BackendCalibrationCli() {
    }

    public static void main(String[] arguments) throws Exception {
        Options options = Options.parse(arguments);
        Path repository = Path.of(System.getProperty("jlc.repository",
            System.getProperty("user.dir", "."))).toAbsolutePath().normalize();
        Path profilePath = options.profilePath();
        if (!profilePath.isAbsolute()) profilePath = repository.resolve(profilePath);

        String previousBackend = System.getProperty(KernelBackendMode.PROPERTY);
        String previousAutotune = System.getProperty(KernelAutotuneMode.PROPERTY);
        String previousLibrary = System.getProperty("jlc.native.lib.path");
        Path temporary = Files.createTempDirectory("jlc-backend-calibration-");
        try {
            // Calibration must measure the R2 Java peer directly, not a profile
            // selected by a previous run in the same JVM.
            System.setProperty(KernelBackendMode.PROPERTY, "r2");
            System.setProperty(KernelAutotuneMode.PROPERTY, "off");
            KernelTuningConfig config = options.tuningConfig();
            List<Workload> workloads = prepareWorkloads(options, temporary.resolve("generated"),
                config);
            if (workloads.isEmpty()) {
                throw new IllegalArgumentException("No calibration workloads were selected");
            }
            buildNativeLibrary(repository, temporary.resolve("native-build"),
                temporary.resolve("generated"));
            Path library = temporary.resolve("native-build").resolve("lib")
                .resolve(System.mapLibraryName("jlc_native"));
            if (!Files.isRegularFile(library)) {
                throw new IOException("Generated JNI library was not produced: " + library);
            }
            System.setProperty("jlc.native.lib.path", library.toString());
            boolean nativeAvailable = NativeGeneratedKernelSupport.isAvailable();
            GeneratedKernelRegistry registry = GeneratedKernelRegistry.global();
            registerArtifacts(registry, workloads);
            KernelMachineIdentity machine = KernelMachineIdentity.current();
            KernelBuildIdentity build = KernelBuildIdentity.current();
            if (!build.identityVerified()) {
                throw new IllegalStateException(
                    "Refusing to persist calibration from an unverified or dirty build; "
                        + "run calibrateBackends from a clean reproducible checkout");
            }
            RuntimeEnvironment environment = RuntimeEnvironment.current(registry, machine, build);
            if (!nativeAvailable) {
                System.out.println("native runtime unavailable; recording the R2 Java peer only");
            }

            KernelCalibrationProfile profile = compatibleExistingProfile(profilePath, machine, build);
            StringBuilder report = new StringBuilder(
                "# JLC typed backend calibration\n"
                    + "# Java and native rows are whole-region execution timings; native rows include JNI\n"
                    + "| Kernel | Workload | R2 Java median ns | Scalar native median ns | AVX2 median ns | Winner |\n"
                    + "|---|---|---:|---:|---:|---|\n");
            for (Workload workload : workloads) {
                BackendCalibrationResult result = calibrate(workload, environment, config);
                profile = profile == null
                    ? KernelCalibrationProfile.fromBackendResult(machine, build, result)
                    : profile.merge(result);
                String row = reportRow(workload, result);
                report.append(row);
            }
            KernelCalibrationProfileStore.save(profilePath, profile);
            System.out.println("profile=" + profilePath);
            System.out.println("entries=" + profile.entries().size());
            System.out.println(report.toString());
        } finally {
            restoreProperty(KernelBackendMode.PROPERTY, previousBackend);
            restoreProperty(KernelAutotuneMode.PROPERTY, previousAutotune);
            restoreProperty("jlc.native.lib.path", previousLibrary);
            deleteTemporary(temporary);
        }
    }

    private static List<Workload> prepareWorkloads(Options options, Path sourceDirectory,
                                                   KernelTuningConfig config)
        throws IOException {
        Files.createDirectories(sourceDirectory);
        List<Workload> result = new ArrayList<>();
        for (int size : options.sizes()) {
            for (int operations : options.operations()) {
                Workload workload = workload(size, operations, config.maxCandidates());
                result.add(workload);
                for (GeneratedKernelSource source : workload.sources()) {
                    Files.writeString(sourceDirectory.resolve(source.symbol() + ".cpp"),
                        source.source(), StandardCharsets.UTF_8);
                }
            }
        }
        return List.copyOf(result);
    }

    private static Workload workload(int size, int operations, int maxCandidates) {
        Matrix a = matrix(size, size, 0.125);
        Matrix b = matrix(size, size, 0.625);
        MatrixExpr expression = MatrixExpr.input(a);
        for (int operation = 0; operation < operations; operation++) {
            expression = (operation & 1) == 0
                ? expression.scale(1.001 + operation * 0.0001)
                : expression.add(MatrixExpr.input(b));
        }
        CompiledMatrixProgram program = MatrixCompiler.compileProgram(
            expression, OptimizationSemantics.STRICT, new FlopCostModel(),
            FusionStrategy.GENERALIZED);
        CpuFusedRegionStep step = program.cpuPlan().steps().stream()
            .filter(CpuFusedRegionStep.class::isInstance)
            .map(CpuFusedRegionStep.class::cast)
            .findFirst().orElseThrow(() -> new IllegalStateException(
                "selected workload did not produce an R2 fused region"));
        KernelLoweringResult lowering = KernelLowerer.lower(step.regionPlan());
        PseudokernelPlan plan = PseudokernelPlanner.plan(lowering.program().function());
        List<GeneratedKernelSource> sources = new ArrayList<>();
        // The Java/scalar/AVX2 race must retain its scalar peer even when a
        // deliberately tiny R5 maxCandidates cap replaces the first
        // generated candidate with the AVX2 baseline.
        if (plan.scalarCppEligible()) {
            sources.add(KernelCodeGenerator.variant(plan,
                KernelVariantSignature.legacyScalar(plan.signature()),
                CppEmissionOptions.nativeRegistry()));
        }
        for (KernelVariantCandidate candidate : KernelVariantGenerator.enumerate(
                plan, OptimizationSemantics.STRICT, maxCandidates)) {
            if (!candidate.eligible() || sources.stream().anyMatch(source ->
                    source.variantSignature().equals(candidate.signature()))) {
                continue;
            }
            sources.add(KernelCodeGenerator.variant(
                plan, candidate.signature(), CppEmissionOptions.nativeRegistry()));
        }
        return new Workload(size, operations, program, lowering, plan, List.copyOf(sources));
    }

    private static void registerArtifacts(GeneratedKernelRegistry registry,
                                          List<Workload> workloads) {
        for (Workload workload : workloads) {
            for (GeneratedKernelSource source : workload.sources()) {
                if (registry.lookup(source.variantSignature()) == null) {
                    registry.registerNative(source.descriptor());
                }
            }
        }
    }

    private static BackendCalibrationResult calibrate(Workload workload,
                                                      RuntimeEnvironment environment,
                                                      KernelTuningConfig config) {
        BackendChoice java = new R2JavaBackend();
        List<BackendCalibrationCandidate> candidates = new ArrayList<>();
        Map<BackendChoice, BackendInvocation> invocations = new HashMap<>();
        List<KernelBinding> correctnessCorpus = KernelValidationCorpus.forPlan(workload.plan());
        candidates.add(BackendCalibrationCandidate.valid(java));
        invocations.put(java, () -> {
            Matrix actual = workload.program().execute();
            consume(actual.getRawData());
            return true;
        });

        BackendChoice scalar = new ScalarNativeBackend(
            KernelVariantSignature.legacyScalar(workload.plan().signature()));
        if (environment.canExecute(workload.plan().signature(), scalar)) {
            String correctness = nativeCorrect(workload, scalar, correctnessCorpus);
            if (correctness == null) {
                candidates.add(BackendCalibrationCandidate.valid(scalar));
                invocations.put(scalar, nativeInvocation(workload, scalar));
            } else {
                candidates.add(BackendCalibrationCandidate.invalid(scalar, correctness));
            }
        }

        List<BackendChoice> avx2 = workload.sources().stream()
            .map(GeneratedKernelSource::variantSignature)
            .filter(variant -> variant.backend() == CodegenBackend.AVX2)
            .map(variant -> (BackendChoice) new GeneratedAvx2Backend(variant))
            .filter(choice -> environment.canExecute(workload.plan().signature(), choice))
            .sorted(Comparator.comparing(choice -> choice.variantOptional()
                .orElseThrow().variantId()))
            .toList();
        for (BackendChoice choice : avx2) {
            String correctness = nativeCorrect(workload, choice, correctnessCorpus);
            if (correctness == null) {
                candidates.add(BackendCalibrationCandidate.valid(choice));
                invocations.put(choice, nativeInvocation(workload, choice));
            } else {
                candidates.add(BackendCalibrationCandidate.invalid(choice, correctness));
            }
        }

        BackendChoice baseline = candidates.stream()
            .map(BackendCalibrationCandidate::choice)
            .filter(choice -> choice instanceof GeneratedAvx2Backend
                && choice.variantOptional().orElseThrow().isBaselineAvx2())
            .findFirst()
            .orElseGet(() -> candidates.stream()
                .map(BackendCalibrationCandidate::choice)
                .filter(choice -> choice instanceof ScalarNativeBackend)
                .findFirst().orElse(java));
        return BackendCalibrationRunner.calibrate(workload.plan().signature(),
            workload.bucket(), baseline, candidates, invocations, config);
    }

    private static String nativeCorrect(Workload workload,
                                        BackendChoice choice,
                                        List<KernelBinding> corpus) {
        KernelCorrectnessResult result = KernelCorrectnessGate.validate(
            workload.plan(),
            binding -> GeneratedKernelExecutor.tryExecuteChoice(
                workload.lowering(), binding, choice, workload.plan()),
            corpus,
            OptimizationSemantics.STRICT);
        return result.passed() ? null : result.diagnostic();
    }

    private static BackendInvocation nativeInvocation(Workload workload,
                                                      BackendChoice choice) {
        return () -> {
            // Match the production fused-region boundary: each invocation
            // receives a fresh heap-real output before binding crosses JNI.
            Matrix output = new Matrix(workload.size(), workload.size());
            boolean invoked = nativeInvoke(workload, choice, output);
            if (invoked) consume(output.getRawData());
            return invoked;
        };
    }

    private static boolean nativeInvoke(Workload workload,
                                         BackendChoice choice,
                                        Matrix output) {
        KernelBinding binding = KernelBinding.fromLogicalBuffers(
            workload.lowering().program().function(), workload.logicalMatrices(output),
            workload.program().cpuPlan().physicalMemoryPlan());
        return GeneratedKernelExecutor.tryExecuteChoice(
            workload.lowering(), binding, choice, workload.plan());
    }

    private static String reportRow(Workload workload, BackendCalibrationResult result) {
        Map<BackendChoice, BackendCandidateEvidence> evidence = new HashMap<>();
        for (BackendCandidateEvidence candidate : result.candidates()) {
            evidence.put(candidate.choice(), candidate);
        }
        BackendChoice java = new R2JavaBackend();
        BackendChoice scalar = new ScalarNativeBackend(
            KernelVariantSignature.legacyScalar(result.kernelSignature()));
        BackendCandidateEvidence avx = result.candidates().stream()
            .filter(candidate -> candidate.choice() instanceof GeneratedAvx2Backend)
            .filter(candidate -> candidate.valid())
            .min(Comparator.comparingDouble(candidate -> candidate.statistics().medianNanos()))
            .orElse(null);
        return "| " + result.kernelSignature().shortHash() + " | " + workload.bucket()
            + " | " + format(evidence.get(java))
            + " | " + format(evidence.get(scalar))
            + " | " + format(avx)
            + " | " + result.selectionStatus().name()
            + (result.winner() == null ? ""
                : ":" + result.winner().tag()
                    + (result.winner().variantOptional().isPresent()
                        ? ":" + result.winner().variantOptional().get().variantId() : ""))
            + " |\n";
    }

    private static String format(BackendCandidateEvidence evidence) {
        if (evidence == null || evidence.statistics() == null) return "n/a";
        String status = evidence.stable() ? "" : ", "
            + evidence.outcome().name().toLowerCase(java.util.Locale.ROOT);
        return String.format("%.1f (%d%s)", evidence.statistics().medianNanos(),
            evidence.statistics().sampleCount(), status);
    }

    private static KernelCalibrationProfile compatibleExistingProfile(Path path,
                                                                      KernelMachineIdentity machine,
                                                                      KernelBuildIdentity build) {
        return KernelCalibrationProfileStore.load(path)
            .filter(profile -> profile.schemaVersion()
                    == KernelCalibrationProfile.CURRENT_SCHEMA_VERSION
                && profile.dispatchSchemaVersion()
                    == KernelCalibrationProfile.CURRENT_DISPATCH_SCHEMA_VERSION
                && KernelCalibrationProfile.METHODOLOGY_VERSION.equals(
                    profile.methodologyVersion())
                && profile.machineKey().equals(machine.key())
                && profile.buildKey().equals(build.key()))
            .orElse(null);
    }

    private static void buildNativeLibrary(Path repository,
                                           Path buildDirectory,
                                           Path sourceDirectory) throws Exception {
        Path javaHome = Path.of(System.getProperty("java.home")).toAbsolutePath();
        run(List.of("cmake", "-S", repository.resolve("native-backend").toString(),
            "-B", buildDirectory.toString(), "-DJLC_JAVA_HOME=" + javaHome,
            "-DJLC_NATIVE_ENABLE_VENDOR_BLAS=OFF",
            "-DJLC_NATIVE_ENABLE_MARCH_NATIVE=OFF",
            "-DJLC_R4_GENERATED_SOURCE_DIR=" + sourceDirectory), repository, 120);
        run(List.of("cmake", "--build", buildDirectory.toString(),
            "--target", "jlc_native", "-j2"), repository, 300);
    }

    private static void run(List<String> command, Path directory, long timeoutSeconds)
        throws Exception {
        Process process = new ProcessBuilder(command).directory(directory.toFile())
            .redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new IOException("Timed out running " + command);
        }
        if (process.exitValue() != 0) {
            throw new IOException("Command failed: " + command + "\n" + output);
        }
    }

    private static Matrix matrix(int rows, int columns, double start) {
        double[] data = new double[rows * columns];
        for (int index = 0; index < data.length; index++) {
            data[index] = start + (index % 97) * 0.001;
        }
        return Matrix.wrap(data, rows, columns);
    }

    private static void consume(double[] values) {
        long value = 0L;
        for (int index = 0; index < values.length; index += Math.max(1, values.length / 16)) {
            value ^= Double.doubleToRawLongBits(values[index]);
        }
        sink ^= value;
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }

    private static void deleteTemporary(Path directory) {
        if (directory == null) return;
        try (Stream<Path> paths = Files.walk(directory)) {
            paths.sorted((left, right) -> right.getNameCount() - left.getNameCount())
                .forEach(path -> {
                    try { Files.deleteIfExists(path); } catch (IOException ignored) { }
                });
        } catch (IOException ignored) {
            // The profile has already been atomically written; a temporary
            // build directory is safe to clean up on the next run if needed.
        }
    }

    private record Workload(int size,
                            int operations,
                            CompiledMatrixProgram program,
                            KernelLoweringResult lowering,
                            PseudokernelPlan plan,
                            List<GeneratedKernelSource> sources) {
        private String bucket() {
            return size + "x" + size + "/ops=" + operations;
        }

        private long elements() {
            return Math.max(0L, (long) size * size);
        }

        private Map<LogicalBuffer, Matrix> logicalMatrices(Matrix output) {
            IdentityHashMap<LogicalBuffer, Matrix> result = new IdentityHashMap<>();
            for (var input : program.cpuPlan().inputBindings()) {
                result.put(input.buffer(), input.matrix());
            }
            LogicalBuffer outputBuffer = lowering.program().function().outputBuffers().get(0)
                .logicalBuffer();
            result.put(outputBuffer, output);
            return result;
        }
    }

    private static final class Options {
        private final List<Integer> sizes;
        private final List<Integer> operations;
        private final Path profilePath;

        private Options(List<Integer> sizes, List<Integer> operations, Path profilePath) {
            this.sizes = sizes;
            this.operations = operations;
            this.profilePath = profilePath;
        }

        private static Options parse(String[] arguments) {
            List<Integer> sizes = parseList(System.getProperty(
                "jlc.compiler.calibration.sizes", "1,4,16,64,256"));
            List<Integer> operations = parseList(System.getProperty(
                "jlc.compiler.calibration.operations", "2,10,25"));
            Path profile = Path.of(System.getProperty("jlc.compiler.autotune.profile",
                "build/profiles/kernel-backends.json"));
            for (int index = 0; index < arguments.length; index++) {
                String argument = arguments[index];
                if (argument.equals("--sizes") || argument.equals("--operations")
                    || argument.equals("--profile")) {
                    if (++index >= arguments.length) {
                        throw new IllegalArgumentException("Missing value for " + argument);
                    }
                    argument += "=" + arguments[index];
                }
                if (argument.startsWith("--sizes=")) sizes = parseList(value(argument));
                else if (argument.startsWith("--operations=")) operations = parseList(value(argument));
                else if (argument.startsWith("--profile=")) profile = Path.of(value(argument));
                else throw new IllegalArgumentException("Unknown calibration argument: " + argument);
            }
            if (sizes.stream().anyMatch(value -> value < 1)
                || operations.stream().anyMatch(value -> value < 1)) {
                throw new IllegalArgumentException("Calibration sizes and operations must be positive");
            }
            return new Options(List.copyOf(sizes), List.copyOf(operations), profile);
        }

        private KernelTuningConfig tuningConfig() {
            return KernelTuningConfig.fromSystemProperties();
        }

        private List<Integer> sizes() { return sizes; }
        private List<Integer> operations() { return operations; }
        private Path profilePath() { return profilePath; }

        private static String value(String argument) {
            return argument.substring(argument.indexOf('=') + 1);
        }

        private static List<Integer> parseList(String text) {
            List<Integer> result = new ArrayList<>();
            for (String value : text.split(",")) {
                if (!value.isBlank()) result.add(Integer.parseInt(value.trim()));
            }
            if (result.isEmpty()) throw new IllegalArgumentException("Calibration list is empty");
            return result;
        }
    }
}
