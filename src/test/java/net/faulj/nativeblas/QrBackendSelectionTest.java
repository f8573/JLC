package net.faulj.nativeblas;

import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;

public class QrBackendSelectionTest {

    @After
    public void cleanup() {
        System.clearProperty("jlc.backend");
        System.clearProperty("jlc.native.qr.provider");
        System.clearProperty("jlc.native.qr.minSize");
        System.clearProperty("jlc.native.qr.decomposeBands");
        System.clearProperty("jlc.native.qr.factorizeBands");
        System.clearProperty("jlc.native.qr.bands");
        System.clearProperty("jlc.native.qr.square.decomposeBands");
        System.clearProperty("jlc.native.qr.square.factorizeBands");
        System.clearProperty("jlc.native.qr.square.bands");
        System.clearProperty("jlc.native.qr.square.decomposeGrid");
        System.clearProperty("jlc.native.qr.square.factorizeGrid");
        System.clearProperty("jlc.native.qr.square.grid");
        System.clearProperty("jlc.native.qr.tall.decomposeBands");
        System.clearProperty("jlc.native.qr.tall.factorizeBands");
        System.clearProperty("jlc.native.qr.tall.bands");
        System.clearProperty("jlc.native.qr.tall.decomposeGrid");
        System.clearProperty("jlc.native.qr.tall.factorizeGrid");
        System.clearProperty("jlc.native.qr.tall.grid");
        System.clearProperty("jlc.native.qr.wide.decomposeBands");
        System.clearProperty("jlc.native.qr.wide.factorizeBands");
        System.clearProperty("jlc.native.qr.wide.bands");
        System.clearProperty("jlc.native.qr.wide.decomposeGrid");
        System.clearProperty("jlc.native.qr.wide.factorizeGrid");
        System.clearProperty("jlc.native.qr.wide.grid");
        System.clearProperty("jlc.native.qr.decomposeGrid");
        System.clearProperty("jlc.native.qr.factorizeGrid");
        System.clearProperty("jlc.native.qr.grid");
        System.clearProperty("jlc.native.qr.calibration.path");
        System.clearProperty("faulj.native.qr.calibration.path");
        NativeFactorizationSupport.resetCalibrationForTests();
        BackendRegistry.resetForTests();
    }

    @Test
    public void autoBandsCanForceJavaForSmallQr() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");
        System.setProperty("jlc.native.qr.decomposeBands", "1-127:java,128+:native");

        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(64, 64, false));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(128, 128, false));
    }

    @Test
    public void autoBandsCanDifferBetweenFactorizeAndDecompose() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");
        System.setProperty("jlc.native.qr.decomposeBands", "1+:native");
        System.setProperty("jlc.native.qr.factorizeBands", "1-255:java,256+:native");

        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(192, 192, false));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(192, 192, true));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(256, 256, true));
    }

    @Test
    public void rectangularFamiliesUseFamilySpecificBands() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");
        System.setProperty("jlc.native.qr.square.decomposeBands", "1+:java");
        System.setProperty("jlc.native.qr.tall.decomposeBands", "1+:native");
        System.setProperty("jlc.native.qr.wide.decomposeBands", "1+:java");

        assertEquals("SQUARE", NativeFactorizationSupport.qrShapeFamilyForTests(192, 192));
        assertEquals("TALL", NativeFactorizationSupport.qrShapeFamilyForTests(512, 128));
        assertEquals("WIDE", NativeFactorizationSupport.qrShapeFamilyForTests(128, 512));

        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(192, 192, false));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(512, 128, false));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(128, 512, false));
    }

    @Test
    public void tallAndWideFactorizeBandsUseShortDimensionMetric() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");
        System.setProperty("jlc.native.qr.tall.factorizeBands", "1-191:native,192+:java");
        System.setProperty("jlc.native.qr.wide.factorizeBands", "1-191:native,192+:java");

        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(512, 128, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(1024, 256, true));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(128, 512, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(256, 1024, true));
    }

    @Test
    public void tallFactorizeGridCanSplitByShortAndLongDimensions() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");
        System.setProperty("jlc.native.qr.tall.factorizeGrid",
            "1-64x1-1024:native,1-64x1025+:java,65-128x1-512:native,65-128x513+:java,129+x1+:java");

        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(1024, 64, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(2048, 64, true));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(512, 128, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(1024, 128, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(1024, 192, true));
    }

    @Test
    public void defaultTallFactorizeGridPrefersJavaBeyondSmallShortDimensionCases() {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");

        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(1024, 64, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(2048, 64, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(512, 128, true));
    }

    @Test
    public void calibrationFileCanDriveQrSelection() throws IOException {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");

        Path calibrationFile = Files.createTempFile("qr-calibration", ".properties");
        Files.writeString(calibrationFile,
            "jlc.native.qr.square.decomposeBands=1-255:java,256+:native" + System.lineSeparator() +
            "jlc.native.qr.tall.factorizeGrid=1-64x1-1024:native,1-64x1025+:java,65+x1+:java" + System.lineSeparator());
        System.setProperty("jlc.native.qr.calibration.path", calibrationFile.toString());
        NativeFactorizationSupport.resetCalibrationForTests();

        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(128, 128, false));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(256, 256, false));
        assertEquals("BUILTIN", NativeFactorizationSupport.qrModeForTests(1024, 64, true));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(2048, 64, true));
    }

    @Test
    public void explicitSystemPropertiesOverrideCalibrationFile() throws IOException {
        assumeNativeBackendReady();
        System.setProperty("jlc.native.qr.provider", "auto");
        System.setProperty("jlc.native.qr.minSize", "1");

        Path calibrationFile = Files.createTempFile("qr-calibration", ".properties");
        Files.writeString(calibrationFile,
            "jlc.native.qr.square.decomposeBands=1+:native" + System.lineSeparator() +
            "jlc.native.qr.tall.factorizeGrid=1-64x1+:native" + System.lineSeparator());
        System.setProperty("jlc.native.qr.calibration.path", calibrationFile.toString());
        System.setProperty("jlc.native.qr.square.decomposeBands", "1+:java");
        System.setProperty("jlc.native.qr.tall.factorizeGrid", "1-64x1+:java");
        NativeFactorizationSupport.resetCalibrationForTests();

        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(256, 256, false));
        assertEquals("DISABLED", NativeFactorizationSupport.qrModeForTests(1024, 64, true));
    }

    private static void assumeNativeBackendReady() {
        System.setProperty("jlc.backend", "native");
        BackendRegistry.resetForTests();
        BackendSnapshot snapshot = BackendRegistry.snapshot();
        Assume.assumeTrue("Native backend unavailable: " + snapshot.nativeContext().getMessage(),
            "native".equals(snapshot.activeBackend()));
    }
}
