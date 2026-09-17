package net.faulj.compiler.matrix.codegen;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.compiler.matrix.kernel.KernelFunction;
import net.faulj.matrix.Matrix;
import net.faulj.nativeblas.NativeGeneratedKernelSupport;

/**
 * Deterministic Java-side view of the generated-kernel registry.
 *
 * <p>Portable R3 objects never contain this registry or a function pointer.
 * Entries are backend artifacts and may be absent; callers must treat a miss
 * as a normal R2 fallback.</p>
 */
public final class GeneratedKernelRegistry {
    private static final GeneratedKernelRegistry GLOBAL = new GeneratedKernelRegistry();

    private final Map<Key, Entry> entries = new ConcurrentHashMap<>();

    public static GeneratedKernelRegistry global() {
        return GLOBAL;
    }

    public void register(GeneratedKernelDescriptor descriptor,
                         GeneratedKernelInvoker invoker) {
        if (descriptor == null || invoker == null) {
            throw new IllegalArgumentException("Generated descriptor and invoker are required");
        }
        Key key = new Key(descriptor.signature(), descriptor.backend());
        Entry previous = entries.putIfAbsent(key, new Entry(descriptor, invoker));
        if (previous != null && !previous.descriptor.symbol().equals(descriptor.symbol())) {
            throw new IllegalArgumentException(
                "Conflicting generated symbol for " + descriptor.signature().shortHash());
        }
    }

    /** Register an artifact that is expected to be present in the native registry. */
    public void registerNative(GeneratedKernelDescriptor descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException("Generated descriptor must not be null");
        }
        register(descriptor, binding -> invokeNative(descriptor, binding));
    }

    public Entry lookup(KernelSignature signature, CodegenBackend backend) {
        if (signature == null || backend == null) {
            return null;
        }
        return entries.get(new Key(signature, backend));
    }

    public int size() {
        return entries.size();
    }

    public void clear() {
        entries.clear();
    }

    private static boolean invokeNative(GeneratedKernelDescriptor descriptor,
                                        KernelBinding binding) {
        KernelFunction function = binding.function();
        double[][] inputs = new double[function.inputBuffers().size()][];
        for (int index = 0; index < inputs.length; index++) {
            Matrix matrix = binding.matrix(function.inputBuffers().get(index));
            if (matrix == null || matrix.getClass() != Matrix.class || matrix.hasImagData()) {
                return false;
            }
            KernelBuffer inputBuffer = function.inputBuffers().get(index);
            if (matrix.getRowCount() != inputBuffer.shape().rows()
                || matrix.getColumnCount() != inputBuffer.shape().columns()) {
                return false;
            }
            inputs[index] = matrix.getRawData();
        }
        KernelBuffer outputBuffer = function.outputBuffers().get(0);
        Matrix output = binding.matrix(outputBuffer);
        if (output == null || output.getClass() != Matrix.class || output.hasImagData()) {
            return false;
        }
        if (output.getRowCount() != outputBuffer.shape().rows()
            || output.getColumnCount() != outputBuffer.shape().columns()) {
            return false;
        }
        for (Matrix input : inputsAsMatrices(function, binding)) {
            if (input == output) {
                return false;
            }
        }
        if (descriptor.backend() == CodegenBackend.AVX2
            && !RuntimeCpuFeatures.avx2Supported()) {
            return false;
        }
        return NativeGeneratedKernelSupport.execute(
            descriptor.signature().canonicalText(), inputs, output.getRawData(),
            output.getRowCount(), output.getColumnCount());
    }

    private static Matrix[] inputsAsMatrices(KernelFunction function, KernelBinding binding) {
        Matrix[] result = new Matrix[function.inputBuffers().size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = binding.matrix(function.inputBuffers().get(index));
        }
        return result;
    }

    public static final class Entry {
        private final GeneratedKernelDescriptor descriptor;
        private final GeneratedKernelInvoker invoker;

        private Entry(GeneratedKernelDescriptor descriptor, GeneratedKernelInvoker invoker) {
            this.descriptor = Objects.requireNonNull(descriptor);
            this.invoker = Objects.requireNonNull(invoker);
        }

        public GeneratedKernelDescriptor descriptor() {
            return descriptor;
        }

        public boolean invoke(KernelBinding binding) {
            return invoker.invoke(binding);
        }
    }

    private record Key(KernelSignature signature, CodegenBackend backend) {
    }
}
