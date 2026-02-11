package net.faulj.autotune;

import net.faulj.autotune.benchmark.CoarseSearch;
import net.faulj.autotune.benchmark.CoarseSearchResult;
import net.faulj.autotune.benchmark.FineSearch;
import net.faulj.autotune.benchmark.FineSearchResult;
import net.faulj.autotune.inference.LimitInferenceEngine;
import net.faulj.autotune.probe.ProbeResult;
import net.faulj.autotune.probe.ProbeRunner;
import net.faulj.autotune.search.SearchSpaceReducer;

import java.util.List;
import java.util.Locale;

/**
 * Demonstrates Phase 1 through Phase 4B of the autotune pipeline.
 *
 * <p>Phase 1: Hardware probes. Phase 2: Conservative limits.
 * Phase 3: Search domain with constraints. Phase 4A: Coarse shape search.
 * Phase 4B: Fine KC + kUnroll refinement.</p>
 */
public final class AutotuneDemo {

    public static void main(String[] args) {
        long wallStart = System.nanoTime();
                // Attempt to load existing profile first (Phase 6 startup validation)
                net.faulj.autotune.persist.MachineFingerprint startupFp = net.faulj.autotune.persist.MachineFingerprint.compute();
                java.util.Optional<net.faulj.autotune.persist.TuningProfile> maybe =
                        net.faulj.autotune.persist.ProfileStore.loadAny();
                if (maybe.isPresent()) {
                        net.faulj.autotune.persist.TuningProfile prof = maybe.get();
                        boolean ok = net.faulj.autotune.persist.ProfileValidator.validate(prof, startupFp, true);
                        if (ok) {
                                net.faulj.autotune.dispatch.BlockSizes bs = new net.faulj.autotune.dispatch.BlockSizes(
                                                prof.mr, prof.nr, prof.kc, prof.kUnroll, prof.mc, prof.nc);
                                net.faulj.autotune.dispatch.TunedDispatchInstaller.install(bs);
                                System.out.println("=== JLC Autotune: Loaded existing valid tuning profile; skipping tuning ===");
                                System.out.println("Profile: " + (new java.io.File(System.getProperty("user.home"), ".jlc/tuning/gemm-" + prof.fingerprint + ".json")).getAbsolutePath());
                                double wallSeconds = (System.nanoTime() - wallStart) / 1e9;
                                System.out.printf(Locale.ROOT, "Total time: %.1fs%n", wallSeconds);
                                return;
                        } else {
                                System.out.println("Existing tuning profile present but failed validation; running tuning pipeline.");
                        }
                }

                // ── Phase 1: Hardware Probes ────────────────────────────────────
                System.out.println("=== JLC Autotune: Phase 1 + 2 + 3 + 4A + 4B ===");
                System.out.println();

                ProbeResult probeResult = ProbeRunner.runAll();

        System.out.println("=== Phase 1 Results ===");
        System.out.println(probeResult);
        System.out.println();

        // ── Phase 2: Limit Inference ────────────────────────────────────
        System.out.println("=== Phase 2: Conservative Limits ===");
        System.out.println();

        LimitInferenceEngine.InferenceResult inferenceResult =
                LimitInferenceEngine.infer(probeResult);

        System.out.println(inferenceResult.limits);
        System.out.println();

        // ── Phase 3: Search Domain ──────────────────────────────────────
        SearchSpaceReducer.ReductionResult reductionResult =
                SearchSpaceReducer.reduce(inferenceResult.limits);

        System.out.println(reductionResult.report);

        // ── Phase 4A: Coarse Search ─────────────────────────────────────
        CoarseSearchResult coarseResult =
                CoarseSearch.run(reductionResult.space);

        System.out.println();
        System.out.println(coarseResult);

        // ── Phase 4B: Fine Search ───────────────────────────────────────
        List<FineSearchResult> fineResults =
                FineSearch.run(coarseResult, reductionResult.space);

        System.out.println("=== Final Results ===");
        for (FineSearchResult fr : fineResults) {
            System.out.println("  " + fr);
        }
        System.out.println();

        // ── Phase 5: Convergence Detection ─────────────────────────────
        System.out.println("=== Phase 5: Convergence Detection ===");
        net.faulj.autotune.converge.ConvergenceCriteria criteria = new net.faulj.autotune.converge.ConvergenceCriteria();
        net.faulj.autotune.converge.ConvergenceReport report =
                net.faulj.autotune.converge.ConvergenceAnalyzer.analyze(fineResults, coarseResult, criteria);

                System.out.println(report);
                System.out.println();

                // ── Phase 6: Persist tuned profile and install dispatch ──────────
                System.out.println("=== Phase 6: Persistence & Dispatch ===");
                try {
                        net.faulj.autotune.persist.MachineFingerprint fp = net.faulj.autotune.persist.MachineFingerprint.compute();
                        java.util.Optional<net.faulj.autotune.persist.TuningProfile> loaded =
                                        net.faulj.autotune.persist.ProfileStore.loadByFingerprint(fp.getHash());

                        boolean installed = false;
                        if (loaded.isPresent()) {
                                net.faulj.autotune.persist.TuningProfile prof = loaded.get();
                                boolean ok = net.faulj.autotune.persist.ProfileValidator.validate(prof, fp, true);
                                if (ok) {
                                        // install into runtime dispatch
                                        net.faulj.autotune.dispatch.BlockSizes bs = new net.faulj.autotune.dispatch.BlockSizes(
                                                        prof.mr, prof.nr, prof.kc, prof.kUnroll, prof.mc, prof.nc);
                                        net.faulj.autotune.dispatch.TunedDispatchInstaller.install(bs);
                                        System.out.println("Loaded existing tuning profile and installed dispatch.");
                                        System.out.println("Profile: " + (new java.io.File(System.getProperty("user.home"), ".jlc/tuning/gemm-" + prof.fingerprint + ".json")).getAbsolutePath());
                                        installed = true;
                                } else {
                                        System.out.println("Existing profile invalid or failed sanity; will persist new profile.");
                                }
                        }

                        if (!installed) {
                                // persist selected best-known config (Phase 5 may not have converged)
                                net.faulj.autotune.benchmark.FineSearchResult sel = report.selected;
                                if (sel != null) {
                                        net.faulj.autotune.persist.TuningProfile prof = net.faulj.autotune.persist.ProfileStore.createFrom(
                                                        fp.getHash(), report);
                                        java.nio.file.Path path = net.faulj.autotune.persist.ProfileStore.save(prof);
                                        if (path != null) {
                                                net.faulj.autotune.dispatch.BlockSizes bs = new net.faulj.autotune.dispatch.BlockSizes(
                                                                prof.mr, prof.nr, prof.kc, prof.kUnroll, prof.mc, prof.nc);
                                                net.faulj.autotune.dispatch.TunedDispatchInstaller.install(bs);
                                                System.out.println("Saved new tuning profile: " + path.toString());
                                        } else {
                                                System.out.println("Failed to save tuning profile; continuing without installing.");
                                        }
                                } else {
                                        System.out.println("No selected configuration found in Phase 5; skipping persistence.");
                                }
                        }
                } catch (Throwable t) {
                        System.err.println("Phase 6 encountered an error: " + t.getMessage());
                }

                // ── Timing ──────────────────────────────────────────────────────
                double wallSeconds = (System.nanoTime() - wallStart) / 1e9;
                System.out.printf(Locale.ROOT, "Total time: %.1fs%n", wallSeconds);
    }
}
