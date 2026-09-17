package net.faulj.compiler.matrix.codegen;

import net.faulj.compiler.matrix.kernel.KernelBinding;
import net.faulj.compiler.matrix.kernel.KernelBuffer;
import net.faulj.matrix.Matrix;

/** End-to-end repeated benchmark for one already compiled variant invocation. */
public final class KernelBenchmark {
    private static volatile long sink;

    private KernelBenchmark() {
    }

    public static KernelBenchmarkStatistics measure(GeneratedKernelInvoker invoker,
                                                    KernelBinding binding,
                                                    KernelTuningConfig config) {
        if (invoker == null || binding == null || config == null) {
            throw new IllegalArgumentException("Benchmark invoker, binding, and config are required");
        }
        for (int warmup = 0; warmup < config.warmupSamples(); warmup++) {
            invokeAndConsume(invoker, binding);
        }
        long started = System.nanoTime();
        long[] samples = new long[config.measuredSamples()];
        for (int sample = 0; sample < samples.length; sample++) {
            long start = System.nanoTime();
            invokeAndConsume(invoker, binding);
            long elapsed = Math.max(1L, System.nanoTime() - start);
            samples[sample] = elapsed;
            if (config.maxBenchmarkMillis() > 0L
                && System.nanoTime() - started >= config.maxBenchmarkMillis() * 1_000_000L) {
                throw new IllegalStateException("R5 benchmark budget exhausted");
            }
        }
        return KernelBenchmarkStatistics.from(samples);
    }

    private static void invokeAndConsume(GeneratedKernelInvoker invoker,
                                         KernelBinding binding) {
        if (!invoker.invoke(binding)) {
            throw new IllegalStateException("Compiled kernel invocation returned false");
        }
        KernelBuffer output = binding.function().outputBuffers().get(0);
        Matrix matrix = binding.matrix(output);
        if (matrix != null) {
            long value = 0L;
            double[] data = matrix.getRawData();
            for (int index = 0; index < data.length; index += Math.max(1, data.length / 16)) {
                value ^= Double.doubleToRawLongBits(data[index]);
            }
            sink ^= value;
        }
    }
}
