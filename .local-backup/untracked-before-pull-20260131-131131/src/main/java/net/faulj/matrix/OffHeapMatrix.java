package net.faulj.matrix;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Matrix backed by an off-heap direct buffer.
 * <p>
 * Note: The current implementation keeps the heap array from {@link Matrix}
 * and provides explicit sync methods for off-heap access. SIMD kernels can
 * use {@link #getDataBuffer()} once wired.
 * Uses 64-byte aligned native memory for SIMD-friendly access.
 * </p>
 */
public class OffHeapMatrix extends Matrix implements AutoCloseable {
    private final Arena arena;
    private final MemorySegment dataSegment;
    private MemorySegment imagSegment;

    public OffHeapMatrix(int rows, int cols) {
        super(rows, cols);
        long bytes = (long) rows * cols * Double.BYTES;
        this.arena = Arena.ofAuto();
        this.dataSegment = arena.allocate(bytes, 64);
    }

    public MemorySegment getDataSegment() {
        return dataSegment;
    }

    public MemorySegment getImagSegment() {
        return imagSegment;
    }

    public boolean isOffHeap() {
        return true;
    }

    public void syncToOffHeap() {
        double[] data = getRawData();
        long offset = 0L;
        for (double v : data) {
            dataSegment.set(ValueLayout.JAVA_DOUBLE, offset, v);
            offset += Double.BYTES;
        }
        double[] imag = getRawImagData();
        if (imag != null) {
            ensureImagSegment();
            long imagOffset = 0L;
            for (double v : imag) {
                imagSegment.set(ValueLayout.JAVA_DOUBLE, imagOffset, v);
                imagOffset += Double.BYTES;
            }
        }
    }

    public void syncFromOffHeap() {
        double[] data = getRawData();
        long offset = 0L;
        for (int i = 0; i < data.length; i++) {
            data[i] = dataSegment.get(ValueLayout.JAVA_DOUBLE, offset);
            offset += Double.BYTES;
        }
        if (imagSegment != null) {
            double[] imag = ensureImagData();
            long imagOffset = 0L;
            for (int i = 0; i < imag.length; i++) {
                imag[i] = imagSegment.get(ValueLayout.JAVA_DOUBLE, imagOffset);
                imagOffset += Double.BYTES;
            }
        }
    }

    private void ensureImagSegment() {
        if (imagSegment == null) {
            long bytes = (long) getRowCount() * getColumnCount() * Double.BYTES;
            imagSegment = arena.allocate(bytes, 64);
        }
    }

    @Override
    public void close() {
        arena.close();
    }
}
