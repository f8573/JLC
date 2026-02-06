package net.faulj.benchmark;

import net.faulj.matrix.Matrix;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.hessenberg.BlockedHessenberg;
import org.junit.Test;

public class QuickHessenbergBenchmark {
    @Test
    public void runQuickBench() {
        int[] sizes = {256, 512, 1024};
        
        System.out.println("Size,Method,Time(ms),GFLOPS");
        
        for (int n : sizes) {
            Matrix a = Matrix.randomMatrix(n, n);
            long flops = (10L * n * n * n) / 3;
            
            // Test with auto-select
            long start = System.nanoTime();
            for (int i = 0; i < 3; i++) {
                HessenbergReduction.decompose(a);
            }
            long end = System.nanoTime();
            double ms = (end - start) / 1e6 / 3;
            double gflops = (flops / (ms / 1000.0)) / 1e9;
            System.out.printf("%d,AutoSelect,%.2f,%.3f\n", n, ms, gflops);
            
            // Test BlockedHessenberg directly
            start = System.nanoTime();
            for (int i = 0; i < 3; i++) {
                BlockedHessenberg.decompose(a);
            }
            end = System.nanoTime();
            ms = (end - start) / 1e6 / 3;
            gflops = (flops / (ms / 1000.0)) / 1e9;
            System.out.printf("%d,Blocked(bs=%d),%.2f,%.3f\n", 
                n, BlockedHessenberg.getBlockSize(), ms, gflops);
        }
    }
}
