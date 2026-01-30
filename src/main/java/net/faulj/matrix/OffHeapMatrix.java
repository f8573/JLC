package net.faulj.matrix;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Off-heap matrix backed by a {@link MemorySegment} with optional alignment.
 * <p>
 * The on-heap backing arrays inherited from {@link Matrix} are kept as a
 * compatibility mirror for existing kernels. Use {@link #syncToOffHeap()} or
 * {@link #syncFromOffHeap()} when crossing between heap and off-heap access.
 * </p>
 */
public final class OffHeapMatrix extends Matrix implements AutoCloseable {
    public enum Order {
        ROW_MAJOR,
        COL_MAJOR
    }

    public static final int DEFAULT_ALIGNMENT = 64;

    private final int rows;
    private final int cols;
    private final Order order;
    private final long ld;
    private final long offsetBytes;
    private final MemorySegment segment;
    private MemorySegment imagSegment;
    private final Arena arena;
    private final boolean ownsArena;
    private final int alignmentBytes;

    /**
     * Create a row-major off-heap matrix with default alignment.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public OffHeapMatrix(int rows, int cols) {
        this(rows, cols, Order.ROW_MAJOR, DEFAULT_ALIGNMENT);
    }

    /**
     * Create an off-heap matrix with explicit layout and alignment.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param order storage order
     * @param alignmentBytes alignment in bytes (power of two)
     */
    public OffHeapMatrix(int rows, int cols, Order order, int alignmentBytes) {
        super(rows, cols);
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative");
        }
        this.rows = rows;
        this.cols = cols;
        this.order = order == null ? Order.ROW_MAJOR : order;
        this.alignmentBytes = normalizeAlignment(alignmentBytes);
        this.ld = (this.order == Order.ROW_MAJOR) ? cols : rows;
        this.offsetBytes = 0L;
        long elements = (this.order == Order.ROW_MAJOR)
            ? (long) rows * ld
            : (long) cols * ld;
        long bytes = Math.multiplyExact(elements, (long) Double.BYTES);
        this.arena = Arena.ofAuto();
        this.segment = arena.allocate(bytes, this.alignmentBytes);
        this.ownsArena = true;
    }

    private OffHeapMatrix(int rows, int cols, Order order, long ld, long offsetBytes,
                          MemorySegment segment, MemorySegment imagSegment,
                          Arena arena, boolean ownsArena, int alignmentBytes) {
        super(rows, cols);
        this.rows = rows;
        this.cols = cols;
        this.order = order == null ? Order.ROW_MAJOR : order;
        this.ld = ld;
        this.offsetBytes = offsetBytes;
        this.segment = segment;
        this.imagSegment = imagSegment;
        this.arena = arena;
        this.ownsArena = ownsArena;
        this.alignmentBytes = normalizeAlignment(alignmentBytes);
    }

    /**
     * Create an off-heap matrix with the specified layout and alignment.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param order storage order
     * @param alignmentBytes alignment in bytes (power of two)
     * @return off-heap matrix
     */
    public static OffHeapMatrix allocate(int rows, int cols, Order order, int alignmentBytes) {
        return new OffHeapMatrix(rows, cols, order, alignmentBytes);
    }

    /**
     * Create a lightweight view on the same off-heap segment.
     *
     * @param rowStart row offset
     * @param colStart column offset
     * @param viewRows view rows
     * @param viewCols view columns
     * @return view matrix
     */
    public OffHeapMatrix view(int rowStart, int colStart, int viewRows, int viewCols) {
        if (rowStart < 0 || colStart < 0 || viewRows < 0 || viewCols < 0) {
            throw new IllegalArgumentException("Invalid view dimensions");
        }
        if (rowStart + viewRows > rows || colStart + viewCols > cols) {
            throw new IllegalArgumentException("View exceeds matrix bounds");
        }
        long newOffset = offsetBytes + elementOffsetBytes(rowStart, colStart, false);
        return new OffHeapMatrix(viewRows, viewCols, order, ld, newOffset,
            segment, imagSegment, arena, false, alignmentBytes);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public Order order() {
        return order;
    }

    public long ld() {
        return ld;
    }

    public long offsetBytes() {
        return offsetBytes;
    }

    public MemorySegment segment() {
        return segment;
    }

    public MemorySegment imagSegment() {
        return imagSegment;
    }

    /**
     * Get a value directly from off-heap storage.
     */
    public double getOffHeap(int row, int col) {
        long byteOffset = elementOffsetBytes(row, col, true);
        return segment.get(ValueLayout.JAVA_DOUBLE, byteOffset);
    }

    /**
     * Set a value directly in off-heap storage.
     */
    public void setOffHeap(int row, int col, double value) {
        long byteOffset = elementOffsetBytes(row, col, true);
        segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, value);
    }

    @Override
    public double get(int row, int column) {
        return super.get(row, column);
    }

    @Override
    public double getImag(int row, int column) {
        return super.getImag(row, column);
    }

    @Override
    public void set(int row, int column, double value) {
        super.set(row, column, value);
        long byteOffset = elementOffsetBytes(row, column, true);
        segment.set(ValueLayout.JAVA_DOUBLE, byteOffset, value);
    }

    @Override
    public void setImag(int row, int column, double value) {
        if (value == 0.0 && getRawImagData() == null) {
            return;
        }
        super.setImag(row, column, value);
        ensureImagSegment();
        long byteOffset = elementOffsetBytes(row, column, true);
        imagSegment.set(ValueLayout.JAVA_DOUBLE, byteOffset, value);
    }

    @Override
    public double[] getRawData() {
        return super.getRawData();
    }

    @Override
    public double[] getRawImagData() {
        return super.getRawImagData();
    }

    @Override
    public double[] ensureImagData() {
        double[] data = super.ensureImagData();
        ensureImagSegment();
        return data;
    }

    @Override
    public void setImagData(double[] imag) {
        super.setImagData(imag);
        if (imag != null) {
            ensureImagSegment();
            syncImagToOffHeap();
        } else {
            imagSegment = null;
        }
    }

    /**
     * Copy the on-heap arrays to off-heap segments.
     */
    public void syncToOffHeap() {
        copyHeapToSegment(getRawData(), segment);
        if (getRawImagData() != null) {
            ensureImagSegment();
            syncImagToOffHeap();
        }
    }

    /**
     * Copy the off-heap segment to the on-heap arrays.
     */
    public void syncFromOffHeap() {
        copySegmentToHeap(segment, getRawData());
        if (imagSegment != null) {
            if (getRawImagData() == null) {
                super.ensureImagData();
            }
            copySegmentToHeap(imagSegment, getRawImagData());
        }
    }

    @Override
    public void close() {
        if (ownsArena && arena != null) {
            arena.close();
        }
    }

    private void ensureImagSegment() {
        if (imagSegment != null) {
            return;
        }
        long elements = (order == Order.ROW_MAJOR)
            ? (long) rows * ld
            : (long) cols * ld;
        long bytes = Math.multiplyExact(elements, (long) Double.BYTES);
        imagSegment = arena.allocate(bytes, alignmentBytes);
        syncImagToOffHeap();
    }

    private void syncImagToOffHeap() {
        double[] imag = getRawImagData();
        if (imag != null) {
            copyHeapToSegment(imag, imagSegment);
        }
    }

    private void copyHeapToSegment(double[] src, MemorySegment dst) {
        if (src == null || dst == null) {
            return;
        }
        if (order == Order.ROW_MAJOR && ld == cols && offsetBytes == 0L) {
            long bytes = Math.multiplyExact((long) src.length, (long) Double.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(src), 0, dst, 0, bytes);
            return;
        }
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                long byteOffset = elementOffsetBytes(r, c, true);
                dst.set(ValueLayout.JAVA_DOUBLE, byteOffset, src[base + c]);
            }
        }
    }

    private void copySegmentToHeap(MemorySegment src, double[] dst) {
        if (src == null || dst == null) {
            return;
        }
        if (order == Order.ROW_MAJOR && ld == cols && offsetBytes == 0L) {
            long bytes = Math.multiplyExact((long) dst.length, (long) Double.BYTES);
            MemorySegment.copy(src, 0, MemorySegment.ofArray(dst), 0, bytes);
            return;
        }
        for (int r = 0; r < rows; r++) {
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                long byteOffset = elementOffsetBytes(r, c, true);
                dst[base + c] = src.get(ValueLayout.JAVA_DOUBLE, byteOffset);
            }
        }
    }

    private long elementOffsetBytes(int row, int col, boolean includeBase) {
        long idx = (order == Order.ROW_MAJOR)
            ? (long) row * ld + col
            : (long) col * ld + row;
        long byteOffset = idx * Double.BYTES;
        return includeBase ? offsetBytes + byteOffset : byteOffset;
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