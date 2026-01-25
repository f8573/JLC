package net.faulj.compute;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import org.junit.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;
import static jcuda.runtime.JCuda.cudaSetDevice;
import jcuda.runtime.cudaDeviceProp;

public class CudaProbeTest {

    @Test
    public void probe() {
        System.out.printf("faulj.cuda.enabled=%s%n", System.getProperty("faulj.cuda.enabled", "not set"));
        CudaSupport.refresh();
        System.out.printf("CudaSupport.isCudaAvailable=%s%n", CudaSupport.isCudaAvailable());
    }
    
    /**
     * Comprehensive test that verifies CUDA is actually being used by:
     * 1. Initializing CUDA runtime and driver
     * 2. Detecting GPU devices and their properties
     * 3. Allocating GPU memory
     * 4. Copying data to GPU
     * 5. Copying data back from GPU
     * 6. Verifying the data transfer worked correctly
     */
    @Test
    public void testCudaActuallyWorking() {
        // Skip test if CUDA is not available
        if (!CudaSupport.isCudaAvailable()) {
            System.out.println("CUDA not available, skipping test");
            return;
        }
        
        System.out.println("=== CUDA Verification Test ===");
        
        try {
            // Step 1: Initialize CUDA Runtime
            System.out.println("\n1. Initializing CUDA Runtime...");
            cudaSetDevice(0);
            System.out.println("   ✓ CUDA Runtime initialized successfully");
            
            // Step 2: Get device properties using runtime API
            System.out.println("\n2. Querying CUDA Device Properties (Runtime API)...");
            cudaDeviceProp prop = new cudaDeviceProp();
            cudaGetDeviceProperties(prop, 0);
            System.out.printf("   Device Name: %s%n", prop.getName());
            System.out.printf("   Compute Capability: %d.%d%n", prop.major, prop.minor);
            System.out.printf("   Total Global Memory: %.2f MB%n", prop.totalGlobalMem / (1024.0 * 1024.0));
            System.out.printf("   Multiprocessor Count: %d%n", prop.multiProcessorCount);
            System.out.printf("   Max Threads Per Block: %d%n", prop.maxThreadsPerBlock);
            System.out.printf("   Warp Size: %d%n", prop.warpSize);
            
            // Step 3: Initialize CUDA Driver API (alternative method)
            System.out.println("\n3. Initializing CUDA Driver API...");
            cuInit(0);
            
            int[] deviceCount = new int[1];
            cuDeviceGetCount(deviceCount);
            System.out.printf("   CUDA Devices Available: %d%n", deviceCount[0]);
            assertTrue("At least one CUDA device should be available", deviceCount[0] > 0);
            
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            
            // Get compute capability using driver API
            int[] major = new int[1];
            int[] minor = new int[1];
            cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            System.out.printf("   Driver API Compute Capability: %d.%d%n", major[0], minor[0]);
            
            byte[] deviceName = new byte[256];
            cuDeviceGetName(deviceName, deviceName.length, device);
            String name = new String(deviceName).trim().replace("\0", "");
            System.out.printf("   Driver API Device Name: %s%n", name);
            
            // Create CUDA context
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);
            System.out.println("   ✓ CUDA Context created successfully");
            
            // Step 4: Allocate and test GPU memory operations
            System.out.println("\n4. Testing GPU Memory Operations...");
            int arraySize = 1024;
            int dataSize = arraySize * Sizeof.FLOAT;
            
            // Create host data
            float[] hostInput = new float[arraySize];
            float[] hostOutput = new float[arraySize];
            for (int i = 0; i < arraySize; i++) {
                hostInput[i] = (float) i;
            }
            System.out.printf("   Created host array with %d elements%n", arraySize);
            
            // Allocate device memory
            CUdeviceptr deviceInput = new CUdeviceptr();
            cuMemAlloc(deviceInput, dataSize);
            System.out.printf("   ✓ Allocated %d bytes on GPU%n", dataSize);
            
            // Copy data to device
            cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), dataSize);
            System.out.println("   ✓ Copied data from host to device");
            
            // Copy data back from device
            cuMemcpyDtoH(Pointer.to(hostOutput), deviceInput, dataSize);
            System.out.println("   ✓ Copied data from device to host");
            
            // Verify data integrity
            System.out.println("\n5. Verifying Data Integrity...");
            boolean dataCorrect = true;
            int errorCount = 0;
            for (int i = 0; i < arraySize && errorCount < 10; i++) {
                if (Math.abs(hostInput[i] - hostOutput[i]) > 1e-5) {
                    System.out.printf("   ✗ Mismatch at index %d: expected %.2f, got %.2f%n", 
                        i, hostInput[i], hostOutput[i]);
                    dataCorrect = false;
                    errorCount++;
                }
            }
            
            if (dataCorrect) {
                System.out.printf("   ✓ All %d elements transferred correctly!%n", arraySize);
            } else {
                System.out.printf("   ✗ Found %s mismatches%n", errorCount >= 10 ? "10+" : errorCount);
            }
            
            assertTrue("Data should transfer correctly through GPU memory", dataCorrect);
            
            // Cleanup
            System.out.println("\n6. Cleaning up...");
            cuMemFree(deviceInput);
            cuCtxDestroy(context);
            System.out.println("   ✓ GPU resources freed");
            
            System.out.println("\n=== CUDA VERIFICATION SUCCESSFUL ===");
            System.out.println("CUDA is properly initialized and functional!");
            
        } catch (Exception e) {
            System.err.println("\n✗ CUDA test failed with exception:");
            e.printStackTrace();
            fail("CUDA operations should work correctly: " + e.getMessage());
        }
    }
    
    /**
     * Simpler test that just verifies we can query basic device info
     */
    @Test
    public void testCudaDeviceQuery() {
        if (!CudaSupport.isCudaAvailable()) {
            System.out.println("CUDA not available, skipping device query test");
            return;
        }
        
        try {
            System.out.println("\n=== Basic CUDA Device Query ===");
            
            // Use runtime API for simplicity
            int[] deviceCount = new int[1];
            cudaGetDeviceCount(deviceCount);
            
            System.out.printf("Found %d CUDA device(s)%n", deviceCount[0]);
            assertTrue("Should find at least one CUDA device", deviceCount[0] > 0);
            
            for (int i = 0; i < deviceCount[0]; i++) {
                cudaDeviceProp prop = new cudaDeviceProp();
                cudaGetDeviceProperties(prop, i);
                
                System.out.printf("\nDevice %d:%n", i);
                System.out.printf("  Name: %s%n", prop.getName());
                System.out.printf("  Compute Capability: %d.%d%n", prop.major, prop.minor);
                System.out.printf("  Total Memory: %.2f GB%n", 
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
                System.out.printf("  Multiprocessors: %d%n", prop.multiProcessorCount);
                System.out.printf("  Max Threads/Block: %d%n", prop.maxThreadsPerBlock);
                System.out.printf("  Clock Rate: %.2f MHz%n", prop.clockRate / 1000.0);
            }
            
            System.out.println("\n✓ Device query successful");
            
        } catch (Exception e) {
            fail("Should be able to query CUDA device properties: " + e.getMessage());
        }
    }
}
