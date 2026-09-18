package net.faulj.autotune.persist;

import net.faulj.autotune.converge.ConvergenceReport;
import org.junit.Test;

import java.nio.file.Files;

import static org.junit.Assert.*;

/**
 * Tests that Phase 5 truth for convergence is persisted exactly.
 */
public class ProfilePersistenceTest {

    @Test
    public void persistsNotConvergedFlagExactly() throws Exception {
        String fp = "unit-test-fp-not-converged";
        // Construct a ConvergenceReport that is NOT converged
        ConvergenceReport report = new ConvergenceReport(false, "unit-test not converged", null,
                0.0, 0.0, false, false, false, false, null, 0.0, 0);

        TuningProfile prof = ProfileStore.createFrom(fp, report);
        assertNotNull(prof);

        java.nio.file.Path path = ProfileStore.save(prof);
        assertNotNull(path);

        String json = new String(Files.readAllBytes(path));
        assertTrue("Persisted JSON must contain converged=false", json.contains("\"converged\":false"));

        // cleanup
        Files.deleteIfExists(path);
    }
}
