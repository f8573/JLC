package net.faulj.compute;

import net.faulj.matrix.Matrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
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

    @Before
    public void resetParallelStateBeforeTest() {
        OptimizedBLAS3.resetParallelStateForTesting();
    }

    @After
    public void resetParallelStateAfterTest() {
        OptimizedBLAS3.resetParallelStateForTesting();
    }

    @Test
    public void actualGemmTilesExecuteInConfiguredPool() {
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        assertEquals(2, GemmDispatch.optimalParallelism(PARALLEL_M, PARALLEL_N, PARALLEL_K, 2, blocks));

        Matrix a = GemmReference.seededMatrix(PARALLEL_M, PARALLEL_K, 101L);
        Matrix b = GemmReference.seededMatrix(PARALLEL_K, PARALLEL_N, 102L);
        Matrix c = new Matrix(PARALLEL_M, PARALLEL_N);
        Set<String> executingThreads = Collections.synchronizedSet(new HashSet<>());
        AtomicBoolean allWorkersDaemon = new AtomicBoolean(true);
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void afterTask(int blockRow, int blockCol) {
                executingThreads.add(Thread.currentThread().getName());
                allWorkersDaemon.compareAndSet(true, Thread.currentThread().isDaemon());
            }
        });

        try {
            OptimizedBLAS3.gemm(a, b, c, 1.0, 0.0, TWO_THREAD_POLICY);
        } finally {
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
        }

        assertFalse("No completed real GEMM tile was observed", executingThreads.isEmpty());
        assertTrue(
            "Expected only configured GEMM workers but observed " + executingThreads,
            executingThreads.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX))
        );
        assertTrue(
            "Common-pool worker executed a GEMM tile: " + executingThreads,
            executingThreads.stream().noneMatch(name -> name.startsWith("ForkJoinPool.commonPool-worker-"))
        );
        assertTrue("Cached GEMM workers must be daemon threads", allWorkersDaemon.get());
    }

    @Test
    public void parallelPolicyMatchesReferenceAcrossSizesAlphaAndBeta() {
        int[] sizes = {64, 137, 512};
        double[] alphas = {1.0, -0.75};
        double[] betas = {0.0, 1.0, 0.5};
        long seed = 7_000L;
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();

        for (int size : sizes) {
            if (size >= 137) {
                assertEquals(GemmDispatch.Kernel.PARALLEL_MICRO,
                    GemmDispatch.selectKernel(size, size, size, false, 2));
                assertEquals(2, GemmDispatch.optimalParallelism(size, size, size, 2, blocks));
            }
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
    public void parallelDispatchReducedToOneThreadAppliesBetaOnce() {
        int m = 256;
        int k = 512;
        int n = 64;
        double alpha = -0.75;
        double beta = 0.5;
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        assertEquals(GemmDispatch.Kernel.PARALLEL_MICRO, GemmDispatch.selectKernel(m, n, k, false, 2));
        assertEquals(1, GemmDispatch.optimalParallelism(m, n, k, 2, blocks));

        Matrix a = GemmReference.seededMatrix(m, k, 8_001L);
        Matrix b = GemmReference.seededMatrix(k, n, 8_002L);
        Matrix initialC = GemmReference.seededMatrix(m, n, 8_003L);
        double[] expected = GemmReference.gemm(
            a.getRawData(), b.getRawData(), initialC.getRawData(), m, k, n, alpha, beta
        );
        Matrix actual = Matrix.wrap(initialC.getRawData().clone(), m, n);

        OptimizedBLAS3.gemm(a, b, actual, alpha, beta, TWO_THREAD_POLICY);

        GemmReference.assertParity(
            "OptimizedBLAS3 effective-single-thread GEMM", m, k, n,
            alpha, beta, 8_000L, expected, actual.getRawData(),
            1e-12, GemmReference.cpuAbsTolerance(expected, k)
        );
    }

    @Test
    public void taskFailurePropagatesWithoutSequentialRetryOrSecondBetaApplication() {
        Matrix a = new Matrix(PARALLEL_M, PARALLEL_K);
        Matrix b = new Matrix(PARALLEL_K, PARALLEL_N);
        Arrays.fill(a.getRawData(), 1.0);
        Arrays.fill(b.getRawData(), 1.0);
        double[] initial = new double[PARALLEL_M * PARALLEL_N];
        Arrays.fill(initial, 8.0);
        Matrix c = Matrix.wrap(initial, PARALLEL_M, PARALLEL_N);
        InjectedTaskFailure injectedFailure = new InjectedTaskFailure("injected after completed tile");
        AtomicBoolean injected = new AtomicBoolean();
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void afterTask(int blockRow, int blockCol) {
                if (blockRow == 0 && blockCol == 0 && injected.compareAndSet(false, true)) {
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
        assertEquals(
            "Completed tile must contain one beta scaling and one alpha*A*B contribution",
            4.0 + PARALLEL_K, c.getRawData()[0], 0.0
        );
    }

    @Test
    public void interruptedWaitRestoresFlagAndPropagatesCause() throws Exception {
        CountDownLatch workerEnteredTask = new CountDownLatch(1);
        CountDownLatch releaseWorkers = new CountDownLatch(1);
        CountDownLatch completedTasks = new CountDownLatch(parallelTaskCount());
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

            @Override
            public void afterTask(int blockRow, int blockCol) {
                completedTasks.countDown();
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
            assertTrue("Interrupted GEMM tasks did not drain", completedTasks.await(10, TimeUnit.SECONDS));
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
            caller.interrupt();
            caller.join(TimeUnit.SECONDS.toMillis(10));
        }
    }

    @Test
    public void excessiveRequestedParallelismIsBoundedByAvailableProcessors() {
        assertEquals(
            Math.max(1, Runtime.getRuntime().availableProcessors()),
            GemmDispatch.boundedParallelism(Integer.MAX_VALUE)
        );

        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        int maxParallelism = GemmDispatch.boundedParallelism(Integer.MAX_VALUE);
        int actual = GemmDispatch.optimalParallelism(
            blocks.mc * Math.max(2, maxParallelism), blocks.nc * 2, blocks.kc,
            Integer.MAX_VALUE, blocks
        );

        assertTrue("Effective parallelism exceeded the processor bound", actual <= maxParallelism);
        assertEquals("Effective parallelism must remain a power of two", actual, Integer.highestOneBit(actual));
    }

    @Test
    public void concurrentSameSizeCallsCreateAndUseOnlyOnePool() throws Exception {
        int callers = 8;
        Set<String> workers = ConcurrentHashMap.newKeySet();
        CyclicBarrier startTogether = new CyclicBarrier(callers);
        CyclicBarrier bothPoolWorkersActive = new CyclicBarrier(2);
        ExecutorService executor = Executors.newFixedThreadPool(callers, runnable -> {
            Thread thread = new Thread(runnable, "gemm-concurrent-caller");
            thread.setDaemon(true);
            return thread;
        });
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void beforeTask(int blockRow, int blockCol) {
                workers.add(Thread.currentThread().getName());
                awaitBarrier(bothPoolWorkersActive);
            }
        });

        try {
            Matrix a = new Matrix(PARALLEL_M, PARALLEL_K);
            Matrix b = new Matrix(PARALLEL_K, PARALLEL_N);
            Set<Future<?>> calls = new HashSet<>();
            for (int i = 0; i < callers; i++) {
                calls.add(executor.submit(() -> {
                    awaitBarrier(startTogether);
                    OptimizedBLAS3.gemm(
                        a, b, new Matrix(PARALLEL_M, PARALLEL_N),
                        1.0, 0.0, TWO_THREAD_POLICY
                    );
                }));
            }
            for (Future<?> call : calls) {
                awaitSuccessful(call);
            }
        } finally {
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
            executor.shutdownNow();
            assertTrue("Concurrent GEMM callers did not terminate", executor.awaitTermination(10, TimeUnit.SECONDS));
        }

        assertEquals("Concurrent same-key lookup must retain one pool", 1, OptimizedBLAS3.sharedPoolCountForTesting());
        assertEquals("A two-thread pool must use exactly two workers", 2, workers.size());
        assertTrue(workers.toString(),
            workers.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX + "p2-")));
    }

    @Test
    public void gemmInvokedFromForkJoinWorkerCompletesInJlcPool() throws Exception {
        Set<String> gemmWorkers = ConcurrentHashMap.newKeySet();
        AtomicReference<String> invokingWorker = new AtomicReference<>();
        ForkJoinPool invokingPool = new ForkJoinPool(1);
        OptimizedBLAS3.setParallelTaskObserverForTesting(new OptimizedBLAS3.ParallelTaskObserver() {
            @Override
            public void afterTask(int blockRow, int blockCol) {
                gemmWorkers.add(Thread.currentThread().getName());
            }
        });

        try {
            invokingPool.submit(() -> {
                invokingWorker.set(Thread.currentThread().getName());
                OptimizedBLAS3.gemm(
                    new Matrix(PARALLEL_M, PARALLEL_K),
                    new Matrix(PARALLEL_K, PARALLEL_N),
                    new Matrix(PARALLEL_M, PARALLEL_N),
                    1.0, 0.0, TWO_THREAD_POLICY
                );
            }).get(30, TimeUnit.SECONDS);
        } finally {
            OptimizedBLAS3.setParallelTaskObserverForTesting(null);
            invokingPool.shutdownNow();
            assertTrue("Invoking ForkJoinPool did not terminate", invokingPool.awaitTermination(10, TimeUnit.SECONDS));
        }

        assertNotNull(invokingWorker.get());
        assertFalse("No completed real GEMM tile was observed", gemmWorkers.isEmpty());
        assertTrue("Nested GEMM escaped the JLC pool: " + gemmWorkers,
            gemmWorkers.stream().allMatch(name -> name.startsWith(GEMM_WORKER_PREFIX)));
        assertFalse("Invoking worker executed a GEMM tile", gemmWorkers.contains(invokingWorker.get()));
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

    private static int parallelTaskCount() {
        GemmDispatch.BlockSizes blocks = GemmDispatch.computeBlockSizes();
        int blockRows = (PARALLEL_M + blocks.mc - 1) / blocks.mc;
        int blockCols = (PARALLEL_N + blocks.nc - 1) / blocks.nc;
        return blockRows * blockCols;
    }

    private static void awaitBarrier(CyclicBarrier barrier) {
        try {
            barrier.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(failure);
        } catch (BrokenBarrierException | TimeoutException failure) {
            throw new RuntimeException(failure);
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
