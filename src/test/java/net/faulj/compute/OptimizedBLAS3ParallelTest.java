package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class OptimizedBLAS3ParallelTest {
    private static final String GEMM_WORKER_PREFIX = "jlc-gemm-worker-";
    private static final DispatchPolicy TWO_THREAD_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(true)
        .parallelism(2)
        .build();

    @Test
    public void actualGemmTilesExecuteInConfiguredPool() throws Exception {
        int m = 256;
        int k = 512;
        int n = 512;
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        assertEquals(2, GemmDispatch.optimalParallelism(m, n, k, 2, blocks));

        Matrix a = GemmReference.seededMatrix(m, k, 101L);
        Matrix b = GemmReference.seededMatrix(k, n, 102L);
        Matrix c = new Matrix(m, n);
        Set<String> executingThreads = Collections.synchronizedSet(new HashSet<>());
        ExecutorService caller = Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, "gemm-test-caller");
            thread.setDaemon(true);
            return thread;
        });

        try {
            Future<?> gemm = caller.submit(() ->
                OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, TWO_THREAD_POLICY)
            );
            captureComputeTileThreadsUntilDone(gemm, executingThreads);
            awaitSuccessful(gemm);
        } finally {
            caller.shutdownNow();
            assertTrue("GEMM caller did not terminate", caller.awaitTermination(10, TimeUnit.SECONDS));
        }

        assertFalse("No thread executing an actual GEMM tile was observed", executingThreads.isEmpty());
        assertTrue(
            "Expected only configured GEMM workers but observed " + executingThreads,
            executingThreads.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX))
        );
        assertTrue(
            "Common-pool worker executed a GEMM tile: " + executingThreads,
            executingThreads.stream().noneMatch(name -> name.startsWith("ForkJoinPool.commonPool-worker-"))
        );
    }

    private static void captureComputeTileThreadsUntilDone(Future<?> gemm, Set<String> executingThreads)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (!gemm.isDone() && System.nanoTime() < deadline) {
            for (Map.Entry<Thread, StackTraceElement[]> entry : Thread.getAllStackTraces().entrySet()) {
                for (StackTraceElement frame : entry.getValue()) {
                    if (frame.getClassName().equals(OptimizedBLAS3.class.getName())
                            && frame.getMethodName().equals("computeTile")) {
                        executingThreads.add(entry.getKey().getName());
                        break;
                    }
                }
            }
            Thread.sleep(1L);
        }
    }

    private static void awaitSuccessful(Future<?> gemm) throws Exception {
        try {
            gemm.get(30, TimeUnit.SECONDS);
        } catch (ExecutionException failure) {
            Throwable cause = failure.getCause();
            if (cause instanceof Exception exception) {
                throw exception;
            }
            throw failure;
        }
    }
}
