package net.faulj.bench;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
@ConditionalOnProperty(prefix = "faulj.benchmark.startup", name = "enabled", havingValue = "true", matchIfMissing = false)
public class StartupBenchmarkRunner implements CommandLineRunner {

    private final BenchmarkService benchmarkService;

    @Autowired
    public StartupBenchmarkRunner(BenchmarkService benchmarkService) {
        this.benchmarkService = benchmarkService;
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("[STARTUP] Running benchmark at startup (StartupBenchmarkRunner)");
        Map<String, Object> res = benchmarkService.runBenchmark();
        Object itObj = res.get("iterations");
        if (itObj instanceof List) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> iterations = (List<Map<String, Object>>) itObj;
            for (Map<String, Object> it : iterations) {
                System.out.printf("[ITER] %s n=%s elapsed=%.6fs gflops=%.6f\n",
                        it.getOrDefault("algorithm", "?"),
                        it.getOrDefault("n", "?"),
                        toDouble(it.get("elapsedSeconds")),
                        toDouble(it.get("gflops")));
            }
        } else {
            System.out.println("[STARTUP] No iterations data returned from benchmark");
        }
    }

    private double toDouble(Object o) {
        if (o == null) return Double.NaN;
        if (o instanceof Number) return ((Number) o).doubleValue();
        try { return Double.parseDouble(o.toString()); } catch (Exception ex) { return Double.NaN; }
    }
}
