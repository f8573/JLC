package net.faulj.web;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;

import net.faulj.core.DiagnosticMetrics;
import net.faulj.core.DiagnosticMetrics.DiagnosticItem;
import net.faulj.core.DiagnosticMetrics.MatrixDiagnostics;
import net.faulj.bench.BenchmarkService;
import net.faulj.eigen.qr.ExplicitQRIteration;
import net.faulj.eigen.schur.RealSchurDecomposition;
import net.faulj.eigen.schur.SchurEigenExtractor;
import net.faulj.matrix.Matrix;
import net.faulj.matrix.MatrixAccuracyValidator;
import net.faulj.scalar.Complex;
import net.faulj.vector.Vector;
import net.faulj.visualizer.MatrixLatexExporter;

@RestController
@CrossOrigin(originPatterns = {
        "http://localhost",
        "http://localhost:*",
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://0.0.0.0",
        "http://0.0.0.0:*",
        "https://localhost",
        "https://localhost:*",
        "https://127.0.0.1",
        "https://127.0.0.1:*",
        "https://0.0.0.0",
        "https://0.0.0.0:*",
        "http://lambdacompute.org",
        "http://lambdacompute.org:*",
        "https://lambdacompute.org",
        "https://lambdacompute.org:*",
        "http://www.lambdacompute.org",
        "https://www.lambdacompute.org"
})
/**
 * REST controller exposing diagnostic and debug endpoints for matrix analysis.
 */
public class ApiController {
    private static final int FULL_DIAGNOSTIC_CELL_LIMIT = 65_536; // e.g. 256x256

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final BenchmarkService benchmarkService;
    // SSE emitters for streaming status updates to connected frontend clients
    private final CopyOnWriteArrayList<SseEmitter> sseEmitters = new CopyOnWriteArrayList<>();
    private volatile String currentStatus = "SERVICE_INTERRUPTION";
    private volatile Map<String, Object> currentCpu = cpuTemplate("offline");
    private final ExecutorService sseExecutor = Executors.newSingleThreadExecutor();
    private final AtomicInteger queuedJobs = new AtomicInteger(0);

    @PreDestroy
    void onShutdown() {
        for (SseEmitter emitter : sseEmitters) {
            try {
                emitter.complete();
            } catch (Exception ignored) {
            }
        }
        sseEmitters.clear();
        sseExecutor.shutdownNow();
    }

    /**
     * Notify connected SSE clients only when the status or CPU state changes.
     */
    private void maybeNotifyStatus(String status, Map<String, Object> cpu) {
        Map<String, Object> nextCpu = new LinkedHashMap<>();
        if (currentCpu != null) {
            nextCpu.putAll(currentCpu);
        } else {
            nextCpu.putAll(cpuTemplate("offline"));
        }
        if (cpu != null) {
            nextCpu.putAll(cpu);
        }

        String cpuState = normalizeCpuState(nextCpu.get("state"));
        nextCpu.put("state", cpuState);
        int q = queuedJobs.get();
        Object queuedObj = nextCpu.get("queuedJobs");
        if (queuedObj instanceof Number n) {
            q = Math.max(0, n.intValue());
        }
        nextCpu.put("queuedJobs", q);

        String derivedStatus = deriveSystemStatus(cpuState, q);
        // Keep optional caller override only when it is one of our known values.
        if (status != null && !status.isBlank()) {
            String normalized = status.trim().toUpperCase();
            if (normalized.equals("ONLINE") || normalized.equals("BUSY")
                    || normalized.equals("LARGE_QUEUE") || normalized.equals("SERVICE_INTERRUPTION")) {
                derivedStatus = normalized;
            }
        }

        boolean changed = false;
        try {
            if (!Objects.equals(derivedStatus, currentStatus)) changed = true;
            else if (!Objects.equals(nextCpu, currentCpu)) changed = true;
        } catch (Exception ex) {
            changed = true;
        }
        if (!changed) return;

        currentStatus = derivedStatus;
        currentCpu = nextCpu;

        // push update to all emitters asynchronously
        sseExecutor.submit(() -> {
            List<SseEmitter> toRemove = new ArrayList<>();
            for (SseEmitter emitter : sseEmitters) {
                try {
                    emitter.send(SseEmitter.event().name("status").data(Map.of("status", currentStatus, "cpu", currentCpu)));
                } catch (Exception ex) {
                    toRemove.add(emitter);
                }
            }
            sseEmitters.removeAll(toRemove);
        });
    }

    private static Map<String, Object> cpuTemplate(String state) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", "CPU");
        m.put("gflops", null);
        m.put("state", state);
        m.put("queuedJobs", 0);
        return m;
    }

    static String normalizeCpuState(Object stateObj) {
        if (stateObj == null) return "offline";
        String s = stateObj.toString().trim().toLowerCase();
        if (s.equals("online") || s.equals("offline") || s.equals("degraded")) {
            return s;
        }
        return "degraded";
    }

    static String deriveSystemStatus(String cpuState, int queued) {
        if (!"online".equals(cpuState)) {
            return "SERVICE_INTERRUPTION";
        }
        if (queued > 10) {
            return "LARGE_QUEUE";
        }
        if (queued > 0) {
            return "BUSY";
        }
        return "ONLINE";
    }

    public ApiController(BenchmarkService benchmarkService) {
        this.benchmarkService = benchmarkService;
        // At startup assume CPU is online and idle.
        maybeNotifyStatus("ONLINE", cpuTemplate("online"));
        // Run a lightweight CPU benchmark in background to estimate GFLOPs
        runCpuBenchmark();
    }

    private void runCpuBenchmark() {
        sseExecutor.submit(() -> {
            try {
                // small-ish size to run quickly but still exercise kernels
                final int n = 300;
                final int iterations = 3;

                // approximate flop counts per run (householder/algorithms):
                // QR (Householder): ~ 2/3 * n^3
                // Hessenberg reduction (Householder): ~ 10/3 * n^3

                List<Double> hessTimes = new ArrayList<>();
                List<Double> qrTimes = new ArrayList<>();

                // Hessenberg test: one warmup then measured iterations, printing per-iteration
                // single warmup run
                {
                    Matrix mw = Matrix.randomMatrix(n, n);
                    long w0 = System.nanoTime();
                    mw.Hessenberg();
                    long w1 = System.nanoTime();
                    double wsec = (w1 - w0) / 1e9;
                    // optionally could log warmup; we skip to stream only per-iteration
                }
                for (int i = 0; i < iterations; i++) {
                    long it0 = System.nanoTime();
                    Matrix m = Matrix.randomMatrix(n, n);
                    m.Hessenberg();
                    long it1 = System.nanoTime();
                    double itSec = (it1 - it0) / 1e9;
                    hessTimes.add(itSec);
                    double flopsPerHess = (10.0 / 3.0) * Math.pow(n, 3);
                    double gflopsIter = flopsPerHess / itSec / 1e9;
                    System.out.printf("[ITER] HESS n=%d run=%d elapsed=%.6fs gflops=%.6f\n", n, i+1, itSec, gflopsIter);
                }
                double secsHess = hessTimes.stream().mapToDouble(Double::doubleValue).sum();
                double flopsPerHess = (10.0 / 3.0) * Math.pow(n, 3);
                double gflopsHess = (flopsPerHess * iterations) / secsHess / 1e9;

                // QR test: one warmup then measured iterations, printing per-iteration
                {
                    Matrix mw = Matrix.randomMatrix(n, n);
                    long w0 = System.nanoTime();
                    mw.QR();
                    long w1 = System.nanoTime();
                    double wsec = (w1 - w0) / 1e9;
                }
                for (int i = 0; i < iterations; i++) {
                    long it0 = System.nanoTime();
                    Matrix m = Matrix.randomMatrix(n, n);
                    m.QR();
                    long it1 = System.nanoTime();
                    double itSec = (it1 - it0) / 1e9;
                    qrTimes.add(itSec);
                    double flopsPerQR = (2.0 / 3.0) * Math.pow(n, 3);
                    double gflopsIter = flopsPerQR / itSec / 1e9;
                    System.out.printf("[ITER] QR n=%d run=%d elapsed=%.6fs gflops=%.6f\n", n, i+1, itSec, gflopsIter);
                }
                double secsQR = qrTimes.stream().mapToDouble(Double::doubleValue).sum();
                double flopsPerQR = (2.0 / 3.0) * Math.pow(n, 3);
                double gflopsQR = (flopsPerQR * iterations) / secsQR / 1e9;

                double avg = (gflopsHess + gflopsQR) / 2.0;

                Map<String, Object> bench = new LinkedHashMap<>();
                bench.put("n", n);
                bench.put("iterations", iterations);
                bench.put("hessenbergTimes", hessTimes);
                bench.put("qrTimes", qrTimes);
                bench.put("secsHess", secsHess);
                bench.put("secsQR", secsQR);
                bench.put("gflopsHess", gflopsHess);
                bench.put("gflopsQR", gflopsQR);
                bench.put("gflopsAvg", avg);

                Map<String, Object> cpu = cpuTemplate("online");
                cpu.put("gflops", avg);
                cpu.put("benchmark", bench);
                maybeNotifyStatus(null, cpu);
            } catch (Throwable ex) {
                // verbose error reporting but avoid crashing startup
                System.err.println("[CPU BENCHMARK] failed: " + ex.getMessage());
                ex.printStackTrace();
                try {
                    Map<String, Object> cpu = cpuTemplate("degraded");
                    cpu.put("gflops", null);
                    Map<String, Object> err = new LinkedHashMap<>();
                    err.put("error", ex.getMessage());
                    cpu.put("benchmarkError", err);
                    maybeNotifyStatus(null, cpu);
                } catch (Exception ignore) {}
            }
        });
    }

    /**
     * Health check endpoint.
     *
     * @return simple status map
     */
    @GetMapping("/api/ping")
    public Map<String, String> ping() {
        return Map.of("message", "pong from Java backend", "version", getServerVersion());
    }

    private String getServerVersion() {
        try {
            String ver = ApiController.class.getPackage().getImplementationVersion();
            if (ver != null && !ver.isBlank()) return ver;
        } catch (Exception ignored) {
        }
        return "development";
    }

    /**
     * Current system status and CPU state used by the sidebar and status dialog.
     */
    @GetMapping("/api/status")
    public Map<String, Object> status() {
        maybeNotifyStatus(null, null);
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("status", currentStatus);
        out.put("cpu", currentCpu);
        return out;
    }

    /**
     * Run Diagnostic512 benchmark flow from backend.
     */
    @GetMapping("/api/benchmark/diagnostic512")
    public ResponseEntity<Map<String, Object>> benchmarkDiagnostic512(
            @RequestParam(value = "iterations", required = false, defaultValue = "5") int iterations) {
        queuedJobs.incrementAndGet();
        maybeNotifyStatus(null, Map.of("state", "online"));
        try {
            Map<String, Object> result = benchmarkService.runDiagnostic512(iterations);
            Object cpuObj = result.get("cpu");
            if (cpuObj instanceof Map<?, ?> cpuMap) {
                @SuppressWarnings("unchecked")
                Map<String, Object> casted = (Map<String, Object>) cpuMap;
                maybeNotifyStatus(null, casted);
            } else {
                maybeNotifyStatus(null, Map.of("state", "online"));
            }
            return ResponseEntity.ok(result);
        } catch (Exception ex) {
            maybeNotifyStatus(null, Map.of("state", "degraded"));
            return ResponseEntity.status(500).body(Map.of("error", "diagnostic512 benchmark failed", "details", ex.getMessage()));
        } finally {
            queuedJobs.updateAndGet(v -> Math.max(0, v - 1));
            maybeNotifyStatus(null, null);
        }
    }

    /**
     * SSE stream that emits status updates only when they change.
     */
    @GetMapping("/api/diagnostics/stream")
    public SseEmitter streamDiagnostics() {
        // Use a finite timeout to avoid long-running servlet async edge-cases
        // (0L = never timeout) — set a large timeout but not infinite.
        SseEmitter emitter = new SseEmitter(300_000L); // 5 minutes
        sseEmitters.add(emitter);

        // Send current status immediately
        try {
            emitter.send(SseEmitter.event().name("status").data(Map.of("status", currentStatus, "cpu", currentCpu)));
        } catch (Exception ignored) {
        }

        emitter.onCompletion(() -> sseEmitters.remove(emitter));
        emitter.onTimeout(() -> sseEmitters.remove(emitter));
        emitter.onError((ex) -> {
            sseEmitters.remove(emitter);
        });
        return emitter;
    }

    /**
     * Generate a sample matrix and return LaTeX for the matrix and Schur factors.
     *
     * @return map containing LaTeX strings and eigenvalue text
     */
    @GetMapping("/api/latex")
    public Map<String, Object> latex() {
        // generate a small random matrix and decompose it
        Matrix m = Matrix.randomMatrix(4, 4);
        Matrix[] mats = ExplicitQRIteration.decompose(m);

        List<String> schurLatex = new ArrayList<>();
        for (Matrix mat : mats) {
            schurLatex.add(MatrixLatexExporter.matrixToLatex(mat));
        }

        // Original matrix latex
        String originalLatex = MatrixLatexExporter.matrixToLatex(m);

        // Eigenvalues
        List<String> eigenStr = new ArrayList<>();
        for (net.faulj.scalar.Complex c : ExplicitQRIteration.getEigenvalues(m.copy())) {
            eigenStr.add(c.toString());
        }

        return Map.of(
                "original", originalLatex,
                "schur", schurLatex,
                "eigenvalues", eigenStr
        );
    }

    /**
     * Compute diagnostics for a matrix posted in the request body.
     *
     * @param payload JSON payload containing a "matrix" field
     * @return diagnostics response
     */
    @PostMapping("/api/diagnostics")
    public ResponseEntity<Map<String, Object>> diagnostics(@RequestBody Map<String, Object> payload) {
        Object matrixObj = payload == null ? null : payload.get("matrix");
        if (matrixObj == null) {
            return ResponseEntity.badRequest().body(Map.of("error", "matrix field is required"));
        }
        queuedJobs.incrementAndGet();
        maybeNotifyStatus(null, Map.of("state", "online"));
        try {
            double[][] data = objectMapper.convertValue(matrixObj, double[][].class);
            if (tooLargeForFullDiagnostics(data)) {
                maybeNotifyStatus(null, Map.of("state", "degraded"));
                return ResponseEntity.unprocessableEntity().body(largeMatrixDiagnosticsHint(data.length, data[0].length));
            }
            Matrix A = new Matrix(data);
            MatrixDiagnostics diagnostics = DiagnosticMetrics.analyze(A);
            maybeNotifyStatus(null, Map.of("state", "online"));
            return ResponseEntity.ok(buildDiagnosticsResponse(diagnostics));
        } catch (Exception ex) {
            maybeNotifyStatus(null, Map.of("state", "degraded"));
            return ResponseEntity.status(500).body(Map.of("error", "diagnostics failed", "details", ex.getMessage()));
        } finally {
            queuedJobs.updateAndGet(v -> Math.max(0, v - 1));
            maybeNotifyStatus(null, null);
        }
    }

    /**
     * Compute diagnostics for a matrix serialized in a query parameter.
     *
     * @param matrixJson matrix data serialized as JSON
     * @return diagnostics response
     */
    @GetMapping("/api/diagnostics")
    public ResponseEntity<Map<String, Object>> diagnosticsGet(@RequestParam(value = "matrix", required = false) String matrixJson) {
        if (matrixJson == null || matrixJson.isBlank()) {
            return ResponseEntity.badRequest().body(Map.of("error", "matrix query param is required"));
        }
        queuedJobs.incrementAndGet();
        maybeNotifyStatus(null, Map.of("state", "online"));
        try {
            double[][] data = objectMapper.readValue(matrixJson, double[][].class);
            if (tooLargeForFullDiagnostics(data)) {
                maybeNotifyStatus(null, Map.of("state", "degraded"));
                return ResponseEntity.unprocessableEntity().body(largeMatrixDiagnosticsHint(data.length, data[0].length));
            }
            Matrix A = new Matrix(data);
            MatrixDiagnostics diagnostics = DiagnosticMetrics.analyze(A);
            maybeNotifyStatus(null, Map.of("state", "online"));
            return ResponseEntity.ok(buildDiagnosticsResponse(diagnostics));
        } catch (JsonProcessingException ex) {
            return ResponseEntity.badRequest().body(Map.of("error", "invalid matrix JSON", "details", ex.getMessage()));
        } catch (Exception ex) {
            maybeNotifyStatus(null, Map.of("state", "degraded"));
            return ResponseEntity.status(500).body(Map.of("error", "diagnostics failed", "details", ex.getMessage()));
        } finally {
            queuedJobs.updateAndGet(v -> Math.max(0, v - 1));
            maybeNotifyStatus(null, null);
        }
    }

    /**
     * Debug endpoint returning raw Schur decomposition artifacts.
     *
     * @param payload JSON payload containing a "matrix" field
     * @return Schur matrices and eigenvalue details
     */
    @PostMapping("/api/debug/schur")
    public ResponseEntity<Map<String, Object>> debugSchur(@RequestBody Map<String, Object> payload) {
        Object matrixObj = payload == null ? null : payload.get("matrix");
        if (matrixObj == null) {
            return ResponseEntity.badRequest().body(Map.of("error", "matrix field is required"));
        }
        double[][] data = objectMapper.convertValue(matrixObj, double[][].class);
        Matrix A = new Matrix(data);

        try {
            net.faulj.decomposition.result.SchurResult schur = RealSchurDecomposition.decompose(A);

            // Re-run extractor on returned T/U to obtain eigenvectors and eigenvalues via same codepath
            SchurEigenExtractor extractor = new SchurEigenExtractor(schur.getT(), schur.getU());

            Map<String, Object> out = new LinkedHashMap<>();
            out.put("t", matrixToMap(schur.getT()));
            out.put("u", matrixToMap(schur.getU()));
            out.put("eigenvalues", toComplexList(extractor.getEigenvalues()));

            // Also provide primitive arrays (real/imag) and block detection
            net.faulj.scalar.Complex[] ev = extractor.getEigenvalues();
            double[] re = new double[ev.length];
            double[] im = new double[ev.length];
            for (int i = 0; i < ev.length; i++) {
                re[i] = ev[i].real;
                im[i] = ev[i].imag;
            }
            out.put("eigenvaluesReal", re);
            out.put("eigenvaluesImag", im);

            // Block detection on T (bottom-up) to show 1x1 vs 2x2 blocks
            Matrix T = schur.getT();
            int n = T.getRowCount();
            double tol = net.faulj.core.Tolerance.get();
            List<Map<String, Object>> blocks = new ArrayList<>();
            int j = n - 1;
            while (j >= 0) {
                if (j == 0 || Math.abs(T.get(j, j - 1)) <= tol) {
                    Map<String, Object> b = new LinkedHashMap<>();
                    b.put("start", j);
                    b.put("end", j);
                    b.put("size", 1);
                    b.put("type", "1x1");
                    blocks.add(0, b);
                    j--;
                } else {
                    int i0 = j - 1;
                    Map<String, Object> b = new LinkedHashMap<>();
                    b.put("start", i0);
                    b.put("end", j);
                    b.put("size", 2);
                    b.put("type", "2x2");
                    blocks.add(0, b);
                    j -= 2;
                }
            }
            out.put("blocks", blocks);

            return ResponseEntity.ok(out);
        } catch (Exception ex) {
            return ResponseEntity.status(500).body(Map.of("error", "exception during schur debug", "details", ex.getMessage()));
        }
    }

    /**
     * Build a nested diagnostics response from computed metrics.
     *
     * @param diagnostics computed diagnostics
     * @return response map
     */
    private Map<String, Object> buildDiagnosticsResponse(MatrixDiagnostics diagnostics) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();

        // 1. Basic Properties
        LinkedHashMap<String, Object> basic = new LinkedHashMap<>();

        LinkedHashMap<String, Object> dimensions = new LinkedHashMap<>();
        dimensions.put("rows", diagnostics.getRows());
        dimensions.put("cols", diagnostics.getCols());
        dimensions.put("columns", diagnostics.getColumns());
        dimensions.put("square", diagnostics.isSquare());
        dimensions.put("real", diagnostics.isReal());
        dimensions.put("domain", diagnostics.getDomain());
        basic.put("dimensionsAndDomain", dimensions);

        LinkedHashMap<String, Object> norms = new LinkedHashMap<>();
        norms.put("norm1", diagnostics.getNorm1());
        norms.put("normInf", diagnostics.getNormInf());
        norms.put("frobeniusNorm", diagnostics.getFrobeniusNorm());
        norms.put("operatorNorm", diagnostics.getOperatorNorm());
        norms.put("density", diagnostics.getDensity());
        norms.put("sparse", diagnostics.getSparse());
        basic.put("normsAndMagnitude", norms);

        LinkedHashMap<String, Object> rank = new LinkedHashMap<>();
        rank.put("rank", diagnostics.getRank());
        rank.put("nullity", diagnostics.getNullity());
        rank.put("fullRank", diagnostics.getFullRank());
        rank.put("rankDeficient", diagnostics.getRankDeficient());
        basic.put("rankAndNullity", rank);

        LinkedHashMap<String, Object> conditioning = new LinkedHashMap<>();
        conditioning.put("invertible", diagnostics.getInvertible());
        conditioning.put("singular", diagnostics.getSingular());
        conditioning.put("leftInvertible", diagnostics.getLeftInvertible());
        conditioning.put("rightInvertible", diagnostics.getRightInvertible());
        conditioning.put("conditionNumber", diagnostics.getConditionNumber());
        conditioning.put("reciprocalConditionNumber", diagnostics.getReciprocalConditionNumber());
        conditioning.put("wellConditioned", diagnostics.getWellConditioned());
        conditioning.put("illConditioned", diagnostics.getIllConditioned());
        conditioning.put("nearlySingular", diagnostics.getNearlySingular());
        basic.put("invertibilityAndConditioning", conditioning);

        LinkedHashMap<String, Object> scalar = new LinkedHashMap<>();
        scalar.put("trace", diagnostics.getTrace());
        scalar.put("determinant", diagnostics.getDeterminant());
        scalar.put("pseudoDeterminant", diagnostics.getPseudoDeterminant());
        scalar.put("psuedoDeterminant", diagnostics.getPseudoDeterminant());
        basic.put("scalarInvariants", scalar);

        LinkedHashMap<String, Object> rowred = new LinkedHashMap<>();
        rowred.put("rowEchelon", diagnostics.getRowEchelon());
        rowred.put("reducedRowEchelon", diagnostics.getReducedRowEchelon());
        basic.put("rowReduction", rowred);

        LinkedHashMap<String, Object> raw = new LinkedHashMap<>();
        raw.put("matrixData", diagnostics.getMatrixData());
        raw.put("matrixImagData", diagnostics.getMatrixImagData());
        basic.put("rawData", raw);

        out.put("basicProperties", basic);

        // 2. Spectral Analysis
        LinkedHashMap<String, Object> spectral = new LinkedHashMap<>();

        LinkedHashMap<String, Object> eigenInfo = new LinkedHashMap<>();
        eigenInfo.put("eigenvalues", toComplexList(diagnostics.getEigenvalues()));
        eigenInfo.put("eigenvectors", matrixToMap(diagnostics.getEigenvectors()));
        eigenInfo.put("eigenspace", matrixToMap(diagnostics.getEigenspace()));

        // Per-eigenvalue aggregated information: value, algebraicMultiplicity, geometricMultiplicity,
        // eigenspace (basis + dimension), representative eigenvector
        List<Map<String, Object>> perEigen = new ArrayList<>();
        List<DiagnosticItem<Set<Vector>>> spaceList = diagnostics.getEigenspaceList();
        List<DiagnosticItem<Set<Vector>>> basisList = diagnostics.getEigenbasisList();
        Complex[] ev = diagnostics.getEigenvalues();
        int[] alg = diagnostics.getAlgebraicMultiplicity();
        int[] geom = diagnostics.getGeometricMultiplicity();
        int groups = spaceList == null ? 0 : spaceList.size();
        for (int i = 0; i < groups; i++) {
            Map<String, Object> info = new LinkedHashMap<>();
            DiagnosticItem<Set<Vector>> spaceItem = spaceList.get(i);
            // determine original eigenvalue index from diagnostic item name (e.g. "eigenspace-2")
            int origIndex = -1;
            if (spaceItem != null && spaceItem.getName() != null && spaceItem.getName().contains("-")) {
                try {
                    String[] parts = spaceItem.getName().split("-");
                    origIndex = Integer.parseInt(parts[parts.length - 1]);
                } catch (Exception ex) {
                    origIndex = -1;
                }
            }
            // eigenvalue
            if (ev != null && origIndex >= 0 && origIndex < ev.length) {
                LinkedHashMap<String, Object> v = new LinkedHashMap<>();
                v.put("real", ev[origIndex].real);
                v.put("imag", ev[origIndex].imag);
                info.put("eigenvalue", v);
            }
            // multiplicities
            info.put("algebraicMultiplicity", alg != null && origIndex >= 0 && origIndex < alg.length ? alg[origIndex] : null);
            info.put("geometricMultiplicity", geom != null && origIndex >= 0 && origIndex < geom.length ? geom[origIndex] : null);
            // dimension is the geometric multiplicity (dimension of eigenspace)
            info.put("dimension", geom != null && origIndex >= 0 && origIndex < geom.length ? geom[origIndex] : null);

            // NOTE: The diagnostic lists were historically labeled incorrectly.
            // Swap: the `eigenspace` diagnostic item actually provides the eigenbasis
            // and the `eigenbasis` diagnostic item provides the eigenspace. Put them
            // into the canonical fields with corrected labels.
            // eigenbasis (from spaceItem)
            Map<String, Object> basisMap = new LinkedHashMap<>();
            if (spaceItem != null && spaceItem.getValue() != null) {
                List<Object> vectors = new ArrayList<>();
                for (Vector v : spaceItem.getValue()) {
                    vectors.add(vectorToPayload(v));
                }
                basisMap.put("vectors", vectors);
                basisMap.put("dimension", spaceItem.getValue().size());
            } else {
                basisMap.put("vectors", null);
                basisMap.put("dimension", 0);
            }
            info.put("eigenbasis", basisMap);

            // eigenspace (from basisList) and representative eigenvector
            if (basisList != null && i < basisList.size()) {
                DiagnosticItem<Set<Vector>> basisItem = basisList.get(i);
                if (basisItem != null && basisItem.getValue() != null && !basisItem.getValue().isEmpty()) {
                    List<Object> spaceVectors = new ArrayList<>();
                    for (Vector v : basisItem.getValue()) spaceVectors.add(vectorToPayload(v));
                    info.put("eigenspace", Map.of("vectors", spaceVectors, "dimension", spaceVectors.size()));
                    // representative is first vector of this eigenspace
                    info.put("representativeEigenvector", spaceVectors.get(0));
                } else {
                    info.put("eigenspace", null);
                    info.put("representativeEigenvector", null);
                }
            } else {
                info.put("eigenspace", null);
                info.put("representativeEigenvector", null);
            }

            perEigen.add(info);
        }
        eigenInfo.put("eigenInformationPerValue", perEigen);
        spectral.put("eigenInformation", eigenInfo);

        LinkedHashMap<String, Object> multiplicity = new LinkedHashMap<>();
        multiplicity.put("algebraicMultiplicity", diagnostics.getAlgebraicMultiplicity());
        multiplicity.put("geometricMultiplicity", diagnostics.getGeometricMultiplicity());
        multiplicity.put("characteristicPolynomial", toComplexList(diagnostics.getCharacteristicPolynomial()));
        multiplicity.put("characteristicPolynomials", toComplexList(diagnostics.getCharacteristicPolynomials()));
        spectral.put("multiplicityAndPolynomials", multiplicity);

        LinkedHashMap<String, Object> spectralVals = new LinkedHashMap<>();
        spectralVals.put("singularValues", diagnostics.getSingularValues());
        spectralVals.put("spectralRadius", diagnostics.getSpectralRadius());
        spectralVals.put("pseudoDeterminant", diagnostics.getPseudoDeterminant());
        spectralVals.put("psuedoDeterminant", diagnostics.getPseudoDeterminant());
        spectral.put("spectralRadiiAndValues", spectralVals);

        LinkedHashMap<String, Object> diagInfo = new LinkedHashMap<>();
        diagInfo.put("defective", diagnostics.getDefective());
        diagInfo.put("diagonalizable", diagnostics.getDiagonalizable());
        spectral.put("diagonalizationAndDefectivity", diagInfo);

        LinkedHashMap<String, Object> normality = new LinkedHashMap<>();
        normality.put("normal", diagnostics.getNormal());
        normality.put("nonNormal", diagnostics.getNonNormal());
        normality.put("orthonormal", diagnostics.getOrthonormal());
        normality.put("orthogonal", diagnostics.getOrthogonal());
        normality.put("unitary", diagnostics.getUnitary());
        spectral.put("normalityAndOrthogonality", normality);

        spectral.put("definiteness", diagnostics.getDefiniteness());
        spectral.put("spectralClass", diagnostics.getSpectralClass());

        out.put("spectralAnalysis", spectral);

        // 3. Structural Properties
        LinkedHashMap<String, Object> structural = new LinkedHashMap<>();

        LinkedHashMap<String, Object> shape = new LinkedHashMap<>();
        shape.put("zero", diagnostics.getZero());
        shape.put("identity", diagnostics.getIdentity());
        shape.put("scalar", diagnostics.getScalar());
        shape.put("diagonal", diagnostics.getDiagonal());
        shape.put("bidiagonal", diagnostics.getBidiagonal());
        shape.put("tridiagonal", diagnostics.getTridiagonal());
        shape.put("upperTriangular", diagnostics.getUpperTriangular());
        shape.put("lowerTriangular", diagnostics.getLowerTriangular());
        shape.put("upperHessenberg", diagnostics.getUpperHessenberg());
        shape.put("lowerHessenberg", diagnostics.getLowerHessenberg());
        shape.put("hessenberg", diagnostics.getHessenberg());
        shape.put("schur", diagnostics.getSchur());
        shape.put("block", diagnostics.getBlock());
        shape.put("companion", diagnostics.getCompanion());
        structural.put("shapeAndCanonicalForms", shape);

        LinkedHashMap<String, Object> symmetry = new LinkedHashMap<>();
        symmetry.put("symmetric", diagnostics.isSymmetric());
        symmetry.put("symmetryError", diagnostics.getSymmetryError());
        symmetry.put("skewSymmetric", diagnostics.getSkewSymmetric());
        symmetry.put("hermitian", diagnostics.getHermitian());
        symmetry.put("persymmetric", diagnostics.getPersymmetric());
        symmetry.put("antidiagonal", diagnostics.getAntidiagonal());
        structural.put("symmetryAndPatterns", symmetry);

        LinkedHashMap<String, Object> algebraic = new LinkedHashMap<>();
        algebraic.put("idempotent", diagnostics.getIdempotent());
        algebraic.put("involutory", diagnostics.getInvolutory());
        algebraic.put("nilpotent", diagnostics.getNilpotent());
        structural.put("algebraicBehavior", algebraic);

        LinkedHashMap<String, Object> geomTransform = new LinkedHashMap<>();
        geomTransform.put("rotation", diagnostics.getRotation());
        geomTransform.put("reflection", diagnostics.getReflection());
        structural.put("geometricTransformations", geomTransform);

        out.put("structuralProperties", structural);

        // 4. Matrix Decompositions
        LinkedHashMap<String, Object> decomp = new LinkedHashMap<>();

        LinkedHashMap<String, Object> primary = new LinkedHashMap<>();
        primary.put("qr", qrToMap(diagnostics.getQr()));
        primary.put("lu", luToMap(diagnostics.getLu()));
        primary.put("cholesky", choleskyToMap(diagnostics.getCholesky()));
        primary.put("svd", svdToMap(diagnostics.getSvd(), diagnostics));
        primary.put("polar", polarToMap(diagnostics.getPolar()));
        decomp.put("primaryDecompositions", primary);

        LinkedHashMap<String, Object> similarity = new LinkedHashMap<>();
        similarity.put("hessenbergDecomposition", hessenbergToMap(diagnostics.getHessenbergDecomposition()));
        similarity.put("schurDecomposition", schurToMap(diagnostics.getSchurDecomposition()));
        similarity.put("diagonalization", diagonalizationToMap(diagnostics.getDiagonalization()));
        similarity.put("symmetricSpectral", spectralToMap(diagnostics.getSymmetricSpectral()));
        similarity.put("bidiagonalization", bidiagonalizationToMap(diagnostics.getBidiagonalization()));
        decomp.put("similarityAndSpectral", similarity);

        LinkedHashMap<String, Object> derived = new LinkedHashMap<>();
        derived.put("inverse", diagnosticItemToMap(diagnostics.getInverse()));
        if (diagnostics.getInverse() != null && diagnostics.getInverse().getValue() != null) {
            derived.put("inverseMatrix", matrixToMap((Matrix) diagnostics.getInverse().getValue()));
        }
        derived.put("pseudoInverse", diagnosticItemToMap(diagnostics.getPseudoInverse()));
        if (diagnostics.getPseudoInverse() != null && diagnostics.getPseudoInverse().getValue() != null) {
            derived.put("pseudoInverseMatrix", matrixToMap((Matrix) diagnostics.getPseudoInverse().getValue()));
        }
        derived.put("pseudoDeterminant", diagnostics.getPseudoDeterminant());
        derived.put("psuedoDeterminant", diagnostics.getPseudoDeterminant());
        derived.put("rref", diagnosticItemToMap(diagnostics.getRref()));
        if (diagnostics.getRref() != null && diagnostics.getRref().getValue() != null) {
            derived.put("rrefMatrix", matrixToMap((Matrix) diagnostics.getRref().getValue()));
        }
        decomp.put("derivedMatrices", derived);

        LinkedHashMap<String, Object> subspaces = new LinkedHashMap<>();
        subspaces.put("rowSpaceBasis", basisToMap(diagnostics.getRowSpaceBasis()));
        subspaces.put("columnSpaceBasis", basisToMap(diagnostics.getColumnSpaceBasis()));
        subspaces.put("nullSpaceBasis", basisToMap(diagnostics.getNullSpaceBasis()));
        decomp.put("subspaceBases", subspaces);

        out.put("matrixDecompositions", decomp);

        return out;
    }

    private static boolean tooLargeForFullDiagnostics(double[][] data) {
        if (data == null || data.length == 0 || data[0] == null) return false;
        int rows = data.length;
        int cols = data[0].length;
        long cells = (long) rows * cols;
        return cells > FULL_DIAGNOSTIC_CELL_LIMIT;
    }

    private static Map<String, Object> largeMatrixDiagnosticsHint(int rows, int cols) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("error", "matrix too large for synchronous full diagnostics");
        out.put("details", "Requested " + rows + "x" + cols + " exceeds limit of " + FULL_DIAGNOSTIC_CELL_LIMIT + " cells");
        out.put("recommendedEndpoint", "/api/benchmark/diagnostic?sizex=512&sizey=512&test=GEMM&iterations=5");
        out.put("reason", "full diagnostics includes many heavy decompositions and large matrix payloads");
        return out;
    }

    /**
     * Generic diagnostic benchmark endpoint supporting sizex/sizey/test parameters.
     * For now this delegates to the existing Diagnostic512 runner when the requested
     * size is 512x512 and test is GEMM. Other combinations are not implemented.
     */
    @GetMapping("/api/benchmark/diagnostic")
    public ResponseEntity<Map<String, Object>> benchmarkDiagnostic(
            @RequestParam(value = "sizex", required = false, defaultValue = "512") int sizex,
            @RequestParam(value = "sizey", required = false, defaultValue = "512") int sizey,
            @RequestParam(value = "test", required = false, defaultValue = "GEMM") String test,
            @RequestParam(value = "iterations", required = false, defaultValue = "5") int iterations) {
        queuedJobs.incrementAndGet();
        maybeNotifyStatus(null, Map.of("state", "online"));
        try {
            // Only 512x512 GEMM is supported by the existing runner implementation.
            if (sizex == 512 && sizey == 512 && (test == null || test.equalsIgnoreCase("GEMM"))) {
                // Run a dedicated GEMM throughput benchmark instead of the full decomposition runner
                Map<String, Object> result = benchmarkService.runGemm(512, iterations);
                Object cpuObj = result.get("cpu");
                if (cpuObj instanceof Map<?, ?> cpuMap) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> casted = (Map<String, Object>) cpuMap;
                    maybeNotifyStatus(null, casted);
                } else {
                    maybeNotifyStatus(null, Map.of("state", "online"));
                }
                return ResponseEntity.ok(result);
            }
            // unsupported combination - guide caller to the 512x512 GEMM endpoint
            return ResponseEntity.status(400).body(Map.of(
                    "error", "unsupported diagnostic benchmark parameters",
                    "details", "Only sizex=512&sizey=512&test=GEMM is supported currently",
                    "recommendedEndpoint", "/api/benchmark/diagnostic?sizex=512&sizey=512&test=GEMM&iterations=5"
            ));
        } catch (Exception ex) {
            maybeNotifyStatus(null, Map.of("state", "degraded"));
            return ResponseEntity.status(500).body(Map.of("error", "diagnostic benchmark failed", "details", ex.getMessage()));
        } finally {
            queuedJobs.updateAndGet(v -> Math.max(0, v - 1));
            maybeNotifyStatus(null, null);
        }
    }

    /**
     * Convert a diagnostic item to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> diagnosticItemToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("status", item.getStatus().name());
        out.put("message", item.getMessage());
        out.put("validation", validationToMap(item.getValidation()));
        return out;
    }

    /**
     * Convert validation results to a response map.
     *
     * @param validation validation result
     * @return mapped response or null
     */
    private Map<String, Object> validationToMap(MatrixAccuracyValidator.ValidationResult validation) {
        if (validation == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("normLevel", validation.normLevel.name());
        out.put("elementLevel", validation.elementLevel.name());
        out.put("normResidual", validation.normResidual);
        out.put("elementResidual", validation.elementResidual);
        out.put("message", validation.message);
        out.put("passes", validation.passes);
        out.put("shouldWarn", validation.shouldWarn);
        return out;
    }

    /**
     * Convert a basis diagnostic item into a vectors response.
     *
     * @param item basis diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> basisToMap(DiagnosticItem<Set<Vector>> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("status", item.getStatus().name());
        out.put("message", item.getMessage());
        if (item.getValue() != null) {
            List<Object> vectors = new ArrayList<>();
            for (Vector v : item.getValue()) {
                vectors.add(vectorToPayload(v));
            }
            out.put("vectors", vectors);
        }
        return out;
    }

    private Object vectorToPayload(Vector v) {
        if (v == null) {
            return null;
        }
        // Always return an array of { real, imag } entries for consistency
        List<Map<String, Object>> entries = new ArrayList<>();
        int n = v.dimension();
        for (int i = 0; i < n; i++) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("real", v.get(i));
            entry.put("imag", v.getImag(i));
            entries.add(entry);
        }
        return entries;
    }

    /**
     * Convert QR decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> qrToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.QRResult qr) {
            out.put("q", matrixToMap(qr.getQ()));
            out.put("r", matrixToMap(qr.getR()));
        }
        return out;
    }

    /**
     * Convert LU decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> luToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.LUResult lu) {
            out.put("l", matrixToMap(lu.getL()));
            out.put("u", matrixToMap(lu.getU()));
            out.put("p", matrixToMap(lu.getP().asMatrix()));
            out.put("singular", lu.isSingular());
            out.put("determinant", lu.getDeterminant());
        }
        return out;
    }

    /**
     * Convert Cholesky decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> choleskyToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.CholeskyResult chol) {
            out.put("l", matrixToMap(chol.getL()));
            out.put("u", matrixToMap(chol.getU()));
            out.put("determinant", chol.getDeterminant());
        }
        return out;
    }

    /**
     * Convert SVD diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> svdToMap(DiagnosticItem<?> item, MatrixDiagnostics diagnostics) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.SVDResult svd) {
            out.put("u", matrixToMap(svd.getU()));
            out.put("sigma", matrixToMap(svd.getSigma()));
            out.put("v", matrixToMap(svd.getV()));
            out.put("singularValues", svd.getSingularValues());
            out.put("rank", svd.getRank(1e-12));
            out.put("conditionNumber", svd.getConditionNumber());
            if (diagnostics != null) {
                out.put("pseudoDeterminant", diagnostics.getPseudoDeterminant());
                out.put("psuedoDeterminant", diagnostics.getPseudoDeterminant());
                if (diagnostics.getPseudoInverse() != null && diagnostics.getPseudoInverse().getValue() != null) {
                    out.put("pseudoInverse", matrixToMap(diagnostics.getPseudoInverse().getValue()));
                }
            }
        }
        return out;
    }

    /**
     * Convert Hessenberg reduction diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> hessenbergToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.HessenbergResult hess) {
            out.put("h", matrixToMap(hess.getH()));
            out.put("q", matrixToMap(hess.getQ()));
        }
        return out;
    }

    /**
     * Convert Schur decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> schurToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.SchurResult schur) {
            out.put("t", matrixToMap(schur.getT()));
            out.put("u", matrixToMap(schur.getU()));
            out.put("eigenvalues", toComplexList(schur.getEigenvalues()));
            out.put("eigenvectors", matrixToMapWithImagFallback(schur.getEigenvectors(), schur.getEigenvalues()));
        }
        return out;
    }

    /**
     * Serialize a matrix but ensure an 'imag' array exists when eigenvalues indicate complex values.
     * This is a best-effort fallback to keep the JSON shape consistent even when the Matrix
     * does not store an imaginary backing array.
     */
    private Map<String, Object> matrixToMapWithImagFallback(Matrix matrix, net.faulj.scalar.Complex[] eigenvalues) {
        Map<String,Object> base = matrixToMap(matrix);
        if (base == null) return null;
        if (base.containsKey("imag")) return base;

        // If there are no complex eigenvalues, nothing to do
        boolean anyComplex = false;
        if (eigenvalues != null) {
            for (net.faulj.scalar.Complex c : eigenvalues) {
                if (Math.abs(c.imag) > 0.0) { anyComplex = true; break; }
            }
        }
        if (!anyComplex) return base;

        // Create zero-filled imag array matching data dimensions
        Object rowsObj = base.get("rows");
        Object colsObj = base.get("cols");
        if (!(rowsObj instanceof Integer) || !(colsObj instanceof Integer)) return base;
        int rows = (Integer) rowsObj;
        int cols = (Integer) colsObj;
        double[][] imag = new double[rows][cols];
        // zeros by default
        base.put("imag", imag);
        return base;
    }

    /**
     * Convert diagonalization diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> diagonalizationToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.eigen.Diagonalization diag) {
            out.put("d", matrixToMap(diag.getD()));
            out.put("p", matrixToMap(diag.getP()));
            out.put("eigenvalues", toComplexList(diag.getEigenvalues()));
            // Compute P inverse
            Matrix p = diag.getP();
            if (p != null) {
                try {
                    Matrix pInverse = net.faulj.inverse.LUInverse.compute(p);
                    out.put("pInverse", matrixToMap(pInverse));
                } catch (Exception e) {
                    // Log the error but still provide structure
                    System.err.println("Failed to compute P inverse: " + e.getMessage());
                    e.printStackTrace();
                    out.put("pInverse", null);
                    out.put("pInverseError", e.getMessage());
                }
            } else {
                out.put("pInverse", null);
            }
        }
        return out;
    }

    /**
     * Convert symmetric spectral decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> spectralToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.symmetric.SpectralDecomposition spectral) {
            out.put("q", matrixToMap(spectral.getEigenvectors()));
            out.put("eigenvalues", spectral.getEigenvalues());
        }
        return out;
    }

    /**
     * Convert bidiagonalization diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> bidiagonalizationToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.BidiagonalizationResult bidiag) {
            out.put("u", matrixToMap(bidiag.getU()));
            out.put("b", matrixToMap(bidiag.getB()));
            out.put("v", matrixToMap(bidiag.getV()));
        }
        return out;
    }

    /**
     * Convert polar decomposition diagnostics to a response map.
     *
     * @param item diagnostic item
     * @return mapped response or null
     */
    private Map<String, Object> polarToMap(DiagnosticItem<?> item) {
        if (item == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>(diagnosticItemToMap(item));
        Object value = item.getValue();
        if (value instanceof net.faulj.decomposition.result.PolarResult polar) {
            out.put("u", matrixToMap(polar.getU()));
            out.put("p", matrixToMap(polar.getP()));
        }
        return out;
    }

    /**
     * Serialize a matrix into a response map with real/imaginary data.
     *
     * @param matrix matrix to serialize
     * @return mapped response or null
     */
    private Map<String, Object> matrixToMap(Matrix matrix) {
        if (matrix == null) {
            return null;
        }
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        int rows = matrix.getRowCount();
        int cols = matrix.getColumnCount();
        out.put("rows", rows);
        out.put("cols", cols);
        double[][] data = new double[rows][cols];
        double[][] imag = matrix.hasImagData() ? new double[rows][cols] : null;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = matrix.get(i, j);
                if (imag != null) {
                    imag[i][j] = matrix.getImag(i, j);
                }
            }
        }
        out.put("data", data);
        if (imag != null) {
            out.put("imag", imag);
        }
        return out;
    }

    /**
     * Convert complex eigenvalues to a list of maps.
     *
     * @param eigenvalues eigenvalues array
     * @return list of real/imag maps or null
     */
    private List<Map<String, Object>> toComplexList(net.faulj.scalar.Complex[] eigenvalues) {
        if (eigenvalues == null) {
            return null;
        }
        List<Map<String, Object>> list = new ArrayList<>();
        for (net.faulj.scalar.Complex c : eigenvalues) {
            LinkedHashMap<String, Object> item = new LinkedHashMap<>();
            item.put("real", c.real);
            item.put("imag", c.imag);
            list.add(item);
        }
        return list;
    }
}
