package net.faulj.nativeblas;

import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class AlgorithmDispatchTest {
    private static final NativeContext READY_CONTEXT = new NativeContext(
        true,
        NativeStatus.READY,
        "test native context",
        "test",
        "test",
        null,
        new NativeMatrixHandle(1L)
    );

    @After
    public void cleanup() {
        System.clearProperty("jlc.algorithm.backend");
        System.clearProperty("jlc.algorithm.gemm.backend");
        System.clearProperty("jlc.algorithm.qr.backend");
        System.clearProperty("jlc.algorithm.calibration.path");
        System.clearProperty("jlc.backend.calibration.path");
        System.clearProperty("jlc.algorithm.calibration.minSamples");
        System.clearProperty("jlc.algorithm.speedupThreshold");
        System.clearProperty("jlc.algorithm.sensitive.speedupThreshold");
        System.clearProperty("jlc.algorithm.gemm.coldStartCppMinSize");
        System.clearProperty("jlc.algorithm.qr.coldStartCppMinSize");
        AlgorithmDispatch.resetForTests();
    }

    @Test
    public void configuredBackendUsesNewAlgorithmProperties() {
        assertEquals(AlgorithmBackend.AUTO, AlgorithmDispatch.configuredBackendForTests("gemm"));

        System.setProperty("jlc.algorithm.backend", "java");
        assertEquals(AlgorithmBackend.JAVA, AlgorithmDispatch.configuredBackendForTests("gemm"));

        System.setProperty("jlc.algorithm.gemm.backend", "cpp");
        assertEquals(AlgorithmBackend.CPP, AlgorithmDispatch.configuredBackendForTests("gemm"));
    }

    @Test
    public void legacyProviderWordsDoNotForceCppAlgorithmBackend() {
        System.setProperty("jlc.algorithm.backend", "native");
        assertEquals(AlgorithmBackend.AUTO, AlgorithmDispatch.configuredBackendForTests("gemm"));

        System.setProperty("jlc.algorithm.gemm.backend", "lapack");
        assertEquals(AlgorithmBackend.AUTO, AlgorithmDispatch.configuredBackendForTests("gemm"));

        System.setProperty("jlc.algorithm.gemm.backend", "builtin");
        assertEquals(AlgorithmBackend.AUTO, AlgorithmDispatch.configuredBackendForTests("gemm"));
    }

    @Test
    public void shapeAndSizeBucketsAreExplicit() {
        assertEquals(ShapeFamily.SQUARE, AlgorithmDispatch.shapeFamilyForTests(128, 128));
        assertEquals(ShapeFamily.TALL, AlgorithmDispatch.shapeFamilyForTests(1024, 128));
        assertEquals(ShapeFamily.WIDE, AlgorithmDispatch.shapeFamilyForTests(128, 1024));

        assertEquals(SizeBand.SMALL, AlgorithmDispatch.sizeBandForTests(128, 64));
        assertEquals(SizeBand.MEDIUM, AlgorithmDispatch.sizeBandForTests(512, 64));
        assertEquals(SizeBand.LARGE, AlgorithmDispatch.sizeBandForTests(1024, 64));
    }

    @Test
    public void coldStartAllowsCppOnlyForFoundationAlgorithmsAboveThreshold() {
        System.setProperty("jlc.algorithm.gemm.coldStartCppMinSize", "256");

        assertFalse(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("gemm", "multiply", 128, 128, 1), READY_CONTEXT));
        assertTrue(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("gemm", "multiply", 256, 256, 1), READY_CONTEXT));
        assertFalse(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("svd", "full", 1024, 1024, 1), READY_CONTEXT));
        assertFalse(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("qr", "decompose_full", 2048, 2048, 1), READY_CONTEXT));
    }

    @Test
    public void calibrationRequiresPassSamplesAndSpeedupThreshold() throws IOException {
        Path profile = Files.createTempFile("algorithm-dispatch", ".properties");
        Files.writeString(profile, String.join(System.lineSeparator(),
            "version=1",
            "bucket.gemm.multiply.square.medium.1.java.samples=5",
            "bucket.gemm.multiply.square.medium.1.java.meanNanos=1100",
            "bucket.gemm.multiply.square.medium.1.cpp.samples=5",
            "bucket.gemm.multiply.square.medium.1.cpp.meanNanos=1000",
            "bucket.gemm.multiply.square.medium.1.cpp.correctness=PASS"));
        System.setProperty("jlc.algorithm.calibration.path", profile.toString());
        AlgorithmDispatch.resetForTests();

        assertTrue(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("gemm", "multiply", 256, 256, 1), READY_CONTEXT));

        System.setProperty("jlc.algorithm.speedupThreshold", "1.20");
        AlgorithmDispatch.resetForTests();
        assertFalse(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("gemm", "multiply", 256, 256, 1), READY_CONTEXT));
    }

    @Test
    public void staleOrFailingCalibrationFallsBackToJava() throws IOException {
        Path profile = Files.createTempFile("algorithm-dispatch-stale", ".properties");
        Files.writeString(profile, String.join(System.lineSeparator(),
            "version=0",
            "bucket.gemm.multiply.square.medium.1.java.samples=5",
            "bucket.gemm.multiply.square.medium.1.java.meanNanos=2000",
            "bucket.gemm.multiply.square.medium.1.cpp.samples=5",
            "bucket.gemm.multiply.square.medium.1.cpp.meanNanos=1000",
            "bucket.gemm.multiply.square.medium.1.cpp.correctness=PASS"));
        System.setProperty("jlc.algorithm.calibration.path", profile.toString());
        AlgorithmDispatch.resetForTests();

        assertTrue("cold-start policy still allows GEMM above threshold",
            AlgorithmDispatch.shouldUseCpp(new AlgorithmDispatchRequest("gemm", "multiply", 256, 256, 1), READY_CONTEXT));

        Files.writeString(profile, String.join(System.lineSeparator(),
            "version=1",
            "bucket.gemm.multiply.square.medium.1.java.samples=5",
            "bucket.gemm.multiply.square.medium.1.java.meanNanos=2000",
            "bucket.gemm.multiply.square.medium.1.cpp.samples=5",
            "bucket.gemm.multiply.square.medium.1.cpp.meanNanos=1000",
            "bucket.gemm.multiply.square.medium.1.cpp.correctness=FAIL"));
        AlgorithmDispatch.resetForTests();

        assertFalse(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("gemm", "multiply", 256, 256, 1), READY_CONTEXT));
    }

    @Test
    public void qrAutoDispatchRequiresPassingCalibrationBucket() throws IOException {
        Path profile = Files.createTempFile("algorithm-dispatch-qr", ".properties");
        Files.writeString(profile, String.join(System.lineSeparator(),
            "version=1",
            "bucket.qr.decompose_thin.tall.medium.1.java.samples=5",
            "bucket.qr.decompose_thin.tall.medium.1.java.meanNanos=3000",
            "bucket.qr.decompose_thin.tall.medium.1.cpp.samples=5",
            "bucket.qr.decompose_thin.tall.medium.1.cpp.meanNanos=2000",
            "bucket.qr.decompose_thin.tall.medium.1.cpp.correctness=PASS",
            "bucket.qr.decompose_full.square.large.1.java.samples=5",
            "bucket.qr.decompose_full.square.large.1.java.meanNanos=3000",
            "bucket.qr.decompose_full.square.large.1.cpp.samples=4",
            "bucket.qr.decompose_full.square.large.1.cpp.meanNanos=1000",
            "bucket.qr.decompose_full.square.large.1.cpp.correctness=PASS",
            "bucket.qr.factorize_only.wide.medium.1.java.samples=5",
            "bucket.qr.factorize_only.wide.medium.1.java.meanNanos=3000",
            "bucket.qr.factorize_only.wide.medium.1.cpp.samples=5",
            "bucket.qr.factorize_only.wide.medium.1.cpp.meanNanos=1000",
            "bucket.qr.factorize_only.wide.medium.1.cpp.correctness=FAIL"));
        System.setProperty("jlc.algorithm.calibration.path", profile.toString());
        AlgorithmDispatch.resetForTests();

        assertTrue(AlgorithmDispatch.shouldUseCpp(
            new AlgorithmDispatchRequest("qr", "decompose_thin", 512, 128, 1), READY_CONTEXT));
        assertFalse("undersampled QR bucket must fall back",
            AlgorithmDispatch.shouldUseCpp(new AlgorithmDispatchRequest("qr", "decompose_full", 1024, 1024, 1), READY_CONTEXT));
        assertFalse("failing QR correctness must fall back",
            AlgorithmDispatch.shouldUseCpp(new AlgorithmDispatchRequest("qr", "factorize_only", 128, 512, 1), READY_CONTEXT));
    }
}
