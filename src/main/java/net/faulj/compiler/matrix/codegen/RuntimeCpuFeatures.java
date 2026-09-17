package net.faulj.compiler.matrix.codegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** Runtime CPU gate used before a generated AVX2 function is invoked. */
public final class RuntimeCpuFeatures {
    private RuntimeCpuFeatures() {
    }

    public static boolean avx2Supported() {
        if (Boolean.getBoolean("jlc.compiler.avx2.disabled")) {
            return false;
        }
        String architecture = System.getProperty("os.arch", "")
            .toLowerCase(Locale.ROOT);
        if (!(architecture.equals("amd64") || architecture.equals("x86_64")
            || architecture.equals("x86"))) {
            return false;
        }
        if ("linux".equalsIgnoreCase(System.getProperty("os.name", ""))) {
            try {
                String cpuInfo = Files.readString(Path.of("/proc/cpuinfo"))
                    .toLowerCase(Locale.ROOT);
                return cpuInfo.lines()
                    .filter(line -> line.startsWith("flags") || line.startsWith("features"))
                    .anyMatch(line -> line.matches(".*(^|\\s)avx2($|\\s).*"));
            } catch (IOException | SecurityException ignored) {
                return false;
            }
        }
        // Unknown operating systems do not get an optimistic AVX2 guess.
        return false;
    }
}
