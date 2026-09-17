package net.faulj.compiler.matrix.kernel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Immutable container for one or more portable kernel functions. */
public final class KernelProgram {
    private final String id;
    private final List<KernelFunction> functions;

    public KernelProgram(String id, List<KernelFunction> functions) {
        this.id = id;
        if (functions == null || functions.stream().anyMatch(Objects::isNull)) {
            throw new IllegalArgumentException("Kernel functions must not be null");
        }
        this.functions = Collections.unmodifiableList(new ArrayList<>(functions));
    }

    public static KernelProgram single(String id, KernelFunction function) {
        return new KernelProgram(id, List.of(function));
    }

    public String id() {
        return id;
    }

    public List<KernelFunction> functions() {
        return functions;
    }

    public int kernelCount() {
        return functions.size();
    }

    /** Return the only function, which is the shape emitted by R3 today. */
    public KernelFunction function() {
        if (functions.size() != 1) {
            throw new IllegalStateException(
                "Expected one kernel function, got " + functions.size());
        }
        return functions.get(0);
    }

    /** Deterministic program dump. */
    public String dump() {
        StringBuilder result = new StringBuilder("kernel-program ").append(id).append('\n');
        if (functions.isEmpty()) {
            result.append("(none)\n");
        } else {
            for (KernelFunction function : functions) {
                result.append(function.dump());
            }
        }
        return result.toString();
    }

    @Override
    public String toString() {
        return dump();
    }
}
