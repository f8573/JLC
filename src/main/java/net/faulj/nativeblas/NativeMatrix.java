package net.faulj.nativeblas;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

/**
 * Native-owned dense matrix storage backed by a direct buffer.
 */
public final class NativeMatrix implements AutoCloseable {
    public enum Order {
        ROW_MAJOR(0),
        COL_MAJOR(1);

        private final int nativeCode;

        Order(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        int nativeCode() {
            return nativeCode;
        }
    }

    public static final int DEFAULT_ALIGNMENT = 64;

    private final int rows;
    private final int cols;
    private final int ld;
    private final int alignmentBytes;
    private final Order order;
    private final long handle;
    private final ByteBuffer buffer;
    private boolean closed;

    private NativeMatrix(int rows, int cols, int ld, int alignmentBytes, Order order, long handle, ByteBuffer buffer) {
        this.rows = rows;
        this.cols = cols;
        this.ld = ld;
        this.alignmentBytes = alignmentBytes;
        this.order = order;
        this.handle = handle;
        this.buffer = buffer;
        this.closed = false;
    }

    public static NativeMatrix allocate(int rows, int cols) {
        return allocate(rows, cols, Order.ROW_MAJOR, DEFAULT_ALIGNMENT);
    }

    public static NativeMatrix allocate(int rows, int cols, Order order, int alignmentBytes) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative");
        }
        Order resolvedOrder = order == null ? Order.ROW_MAJOR : order;
        int normalizedAlignment = normalizeAlignment(alignmentBytes);
        long handle = NativeBindings.nativeMatrixCreate(rows, cols, resolvedOrder.nativeCode(), normalizedAlignment);
        if (handle == 0L) {
            throw new OutOfMemoryError("Failed to allocate native matrix buffer");
        }
        ByteBuffer buffer = NativeBindings.nativeMatrixBuffer(handle);
        if (buffer == null) {
            NativeBindings.nativeMatrixDestroy(handle);
            throw new IllegalStateException("Failed to map native matrix buffer");
        }
        buffer.order(ByteOrder.nativeOrder());
        int ld = resolvedOrder == Order.ROW_MAJOR ? cols : rows;
        return new NativeMatrix(rows, cols, ld, normalizedAlignment, resolvedOrder, handle, buffer);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public int ld() {
        return ld;
    }

    public int alignmentBytes() {
        return alignmentBytes;
    }

    public Order order() {
        return order;
    }

    public long handle() {
        return handle;
    }

    public ByteBuffer buffer() {
        return buffer;
    }

    public DoubleBuffer asDoubleBuffer() {
        return buffer.asDoubleBuffer();
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        NativeBindings.nativeMatrixDestroy(handle);
        closed = true;
    }

    private static int normalizeAlignment(int alignmentBytes) {
        if (alignmentBytes <= 0) {
            return DEFAULT_ALIGNMENT;
        }
        int value = 1;
        while (value < alignmentBytes && value > 0) {
            value <<= 1;
        }
        return value <= 0 ? DEFAULT_ALIGNMENT : value;
    }
}
