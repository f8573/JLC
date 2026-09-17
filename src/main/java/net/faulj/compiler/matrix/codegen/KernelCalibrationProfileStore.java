package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;

/** Explicit-path profile persistence; no normal execution writes to user home. */
public final class KernelCalibrationProfileStore {
    private KernelCalibrationProfileStore() {
    }

    public static void save(Path path, KernelCalibrationProfile profile) throws IOException {
        if (path == null || profile == null) {
            throw new IllegalArgumentException("Profile path and profile are required");
        }
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        Path absolute = path.toAbsolutePath();
        Path temporary = Files.createTempFile(
            absolute.getParent(), absolute.getFileName().toString(), ".tmp");
        try {
            Files.writeString(temporary, profile.toJson(), StandardCharsets.UTF_8);
            try {
                Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
            } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public static Optional<KernelCalibrationProfile> load(Path path) {
        if (path == null || !Files.isRegularFile(path)) return Optional.empty();
        try {
            return Optional.of(KernelCalibrationProfile.fromJson(
                Files.readString(path, StandardCharsets.UTF_8)));
        } catch (IOException | RuntimeException ignored) {
            return Optional.empty();
        }
    }

    public static Optional<KernelCalibrationProfile> loadConfigured() {
        String configured = System.getProperty("jlc.compiler.autotune.profile");
        return configured == null || configured.isBlank()
            ? Optional.empty() : load(Path.of(configured));
    }
}
