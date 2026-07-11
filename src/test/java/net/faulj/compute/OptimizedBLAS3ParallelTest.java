package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class OptimizedBLAS3ParallelTest {
    private static final String GEMM_WORKER_PREFIX = "jlc-gemm-worker-";
    private static final int PARALLEL_M = 256;
    private static final int PARALLEL_K = 512;
    private static final int PARALLEL_N = 512;
    private static final DispatchPolicy TWO_THREAD_POLICY = DispatchPolicy.builder()
        .enableCuda(false)
        .enableParallel(true)
        .parallelism(2)
        .build();

    @Test
    public void actualGemmTilesExecuteInConfiguredPool() throws Exception {
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        assertEquals(2, GemmDispatch.optimalParallelism(PARALLEL_M, PARALLEL_N, PARALLEL_K, 2, blocks));

        Matrix a = GemmReference.seededMatrix(PARALLEL_M, PARALLEL_K, 101L);
        Matrix b = GemmReference.seededMatrix(PARALLEL_K, PARALLEL_N, 102L);
        Matrix c = new Matrix(PARALLEL_M, PARALLEL_N);
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

    @Test
    public void parallelPolicyMatchesReferenceAcrossSizesAlphaAndBeta() {
        int[] sizes = {64, 127, 512};
        double[] alphas = {1.0, -0.75};
        double[] betas = {0.0, 1.0, 0.5};
        long seed = 7_000L;

        for (int size : sizes) {
            for (double alpha : alphas) {
                for (double beta : betas) {
                    Matrix a = GemmReference.seededMatrix(size, size, seed + 1);
                    Matrix b = GemmReference.seededMatrix(size, size, seed + 2);
                    Matrix initialC = GemmReference.seededMatrix(size, size, seed + 3);
                    double[] expected = GemmReference.gemm(
                        a.getRawData(), b.getRawData(), initialC.getRawData(),
                        size, size, size, alpha, beta
                    );
                    Matrix actual = Matrix.wrap(initialC.getRawData().clone(), size, size);

                    OptimizedBLAS3.gemm(a, b, actual, alpha, beta, TWO_THREAD_POLICY);

                    GemmReference.assertParity(
                        "OptimizedBLAS3 parallel-policy GEMM", size, size, size,
                        alpha, beta, seed, expected, actual.getRawData(),
                        1e-12, GemmReference.cpuAbsTolerance(expected, size)
                    );
                    seed += 17;
                }
            }
        }
    }

    @Test
    public void taskFailurePropagatesWithoutSequentialRetryOrSecondBetaApplication() {
        Matrix a = new Matrix(PARALLEL_M, PARALLEL_K);
        Matrix b = new Matrix(PARALLEL_K, PARALLEL_N);
        double[] initial = new double[PARALLEL_M * PARALLEL_N];
        Arrays.fill(initial, 8.0);
        Matrix c = Matrix.wrap(initial, PARALLEL_M, PARALLEL_N);
        InjectedTaskFailure injectedFailure = new InjectedTaskFailure("injected after completed tile");
        AtomicBoolean injected = new AtomicBoolean();
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void afterTask(int blockRow, int blockCol) {
                if (injected.compareAndSet(false, true)) {
                    throw injectedFailure;
                }
            }
        });

        try {
            OptimizedBLAS3.gemm(a, b, c, 1.0, 0.5, TWO_THREAD_POLICY);
            fail("Parallel GEMM task failure must reach the caller");
        } catch (RuntimeException failure) {
            assertEquals("parallel GEMM failed", failure.getMessage());
            assertSame("Injected cause must remain inspectable", injectedFailure, findCause(failure, InjectedTaskFailure.class));
        } finally {
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
        }

        assertTrue("Failure was not injected after parallel work completed", injected.get());
        for (double value : c.getRawData()) {
            assertEquals("beta must be applied once and no sequential retry may run", 4.0, value, 0.0);
        }
    }

    @Test
    public void interruptedWaitRestoresFlagAndPropagatesCause() throws Exception {
        CountDownLatch workerEnteredTask = new CountDownLatch(1);
        CountDownLatch releaseWorkers = new CountDownLatch(1);
        AtomicReference<Throwable> observedFailure = new AtomicReference<>();
        AtomicBoolean interruptFlagObserved = new AtomicBoolean();
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void beforeTask(int blockRow, int blockCol) {
                workerEnteredTask.countDown();
                try {
                    if (!releaseWorkers.await(20, TimeUnit.SECONDS)) {
                        throw new RuntimeException("timed out waiting to release GEMM worker");
                    }
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(failure);
                }
            }
        });

        Thread caller = new Thread(() -> {
            try {
                OptimizedBLAS3.gemm(
                    new Matrix(PARALLEL_M, PARALLEL_K),
                    new Matrix(PARALLEL_K, PARALLEL_N),
                    new Matrix(PARALLEL_M, PARALLEL_N),
                    1.0, 0.5, TWO_THREAD_POLICY
                );
            } catch (Throwable failure) {
                observedFailure.set(failure);
                interruptFlagObserved.set(Thread.currentThread().isInterrupted());
            }
        }, "gemm-interrupted-caller");

        try {
            caller.start();
            assertTrue("No GEMM worker entered a real task", workerEnteredTask.await(10, TimeUnit.SECONDS));
            caller.interrupt();
            caller.join(TimeUnit.SECONDS.toMillis(10));
            assertFalse("Interrupted GEMM caller did not terminate", caller.isAlive());

            Throwable failure = observedFailure.get();
            assertNotNull("Interrupted GEMM call returned successfully", failure);
            assertEquals("parallel GEMM failed", failure.getMessage());
            assertNotNull("InterruptedException cause must remain inspectable", findCause(failure, InterruptedException.class));
            assertTrue("Caller interrupt flag was not restored", interruptFlagObserved.get());
        } finally {
            releaseWorkers.countDown();
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
            caller.interrupt();
            caller.join(TimeUnit.SECONDS.toMillis(10));
        }
    }

    @Test
    public void poolIsReusedAndCachedByEffectiveParallelism() {
        Set<String> firstTwoThreadWorkers = executeWithAllWorkersObserved(2);
        Set<String> secondTwoThreadWorkers = executeWithAllWorkersObserved(2);
        Set<String> fourThreadWorkers = executeWithAllWorkersObserved(4);
        Set<String> finalTwoThreadWorkers = executeWithAllWorkersObserved(2);

        assertEquals("Two-thread pool should expose two workers", 2, firstTwoThreadWorkers.size());
        assertEquals("Repeated calls should reuse the same two-thread workers", firstTwoThreadWorkers, secondTwoThreadWorkers);
        assertEquals("Two-thread pool should survive use of another effective size", firstTwoThreadWorkers, finalTwoThreadWorkers);
        assertEquals("Four-thread pool should expose four workers", 4, fourThreadWorkers.size());
        assertTrue(firstTwoThreadWorkers.toString(),
            firstTwoThreadWorkers.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX + "p2-")));
        assertTrue(fourThreadWorkers.toString(),
            fourThreadWorkers.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX + "p4-")));
        assertTrue("Pools of different effective sizes must use distinct workers",
            Collections.disjoint(firstTwoThreadWorkers, fourThreadWorkers));
        assertTrue("No GEMM task may use the common pool",
            firstTwoThreadWorkers.stream().noneMatch(name -> name.contains("commonPool"))
                && fourThreadWorkers.stream().noneMatch(name -> name.contains("commonPool")));
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

    private static Set<String> executeWithAllWorkersObserved(int requestedParallelism) {
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        assertEquals(requestedParallelism,
            GemmDispatch.optimalParallelism(PARALLEL_M, PARALLEL_N, PARALLEL_K, requestedParallelism, blocks));
        Set<String> workers = ConcurrentHashMap.newKeySet();
        CyclicBarrier allWorkersActive = new CyclicBarrier(requestedParallelism);
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void beforeTask(int blockRow, int blockCol) {
                workers.add(Thread.currentThread().getName());
                try {
                    allWorkersActive.await(10, TimeUnit.SECONDS);
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(failure);
                } catch (BrokenBarrierException | TimeoutException failure) {
                    throw new RuntimeException(failure);
                }
            }
        });

        try {
            DispatchPolicy policy = DispatchPolicy.builder()
                .enableCuda(false)
                .enableParallel(true)
                .parallelism(requestedParallelism)
                .build();
            OptimizedBLAS3.gemm(
                new Matrix(PARALLEL_M, PARALLEL_K),
                new Matrix(PARALLEL_K, PARALLEL_N),
                new Matrix(PARALLEL_M, PARALLEL_N),
                1.0, 0.0, policy
            );
        } finally {
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
        }
        return new HashSet<>(workers);
    }

    private static <T extends Throwable> T findCause(Throwable failure, Class<T> type) {
        Throwable current = failure;
        while (current != null) {
            if (type.isInstance(current)) {
                return type.cast(current);
            }
            current = current.getCause();
        }
        return null;
    }

    private static final class InjectedTaskFailure extends RuntimeException {
        private InjectedTaskFailure(String message) {
            super(message);
        }
    }
}
