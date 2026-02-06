package net.faulj.web;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class ApiControllerStatusTest {

    @Test
    public void offlineAndDegradedCpuMapToServiceInterruption() {
        assertEquals("SERVICE_INTERRUPTION", ApiController.deriveSystemStatus("offline", 0));
        assertEquals("SERVICE_INTERRUPTION", ApiController.deriveSystemStatus("offline", 99));
        assertEquals("SERVICE_INTERRUPTION", ApiController.deriveSystemStatus("degraded", 0));
        assertEquals("SERVICE_INTERRUPTION", ApiController.deriveSystemStatus("degraded", 7));
    }

    @Test
    public void onlineCpuWithNoQueueIsOnline() {
        assertEquals("ONLINE", ApiController.deriveSystemStatus("online", 0));
    }

    @Test
    public void onlineCpuWithQueueUpToTenIsBusy() {
        assertEquals("BUSY", ApiController.deriveSystemStatus("online", 1));
        assertEquals("BUSY", ApiController.deriveSystemStatus("online", 10));
    }

    @Test
    public void onlineCpuWithQueueAboveTenIsLargeQueue() {
        assertEquals("LARGE_QUEUE", ApiController.deriveSystemStatus("online", 11));
        assertEquals("LARGE_QUEUE", ApiController.deriveSystemStatus("online", 100));
    }

    @Test
    public void cpuStateNormalizationMatchesAllowedValues() {
        assertEquals("online", ApiController.normalizeCpuState("online"));
        assertEquals("offline", ApiController.normalizeCpuState("offline"));
        assertEquals("degraded", ApiController.normalizeCpuState("degraded"));
        assertEquals("degraded", ApiController.normalizeCpuState("unexpected"));
        assertEquals("offline", ApiController.normalizeCpuState(null));
    }
}
