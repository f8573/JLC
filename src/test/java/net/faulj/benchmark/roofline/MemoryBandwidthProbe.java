package net.faulj.benchmark.roofline;

final class MemoryBandwidthProbe {
    private static final String GLOBAL_BANDWIDTH_PROPERTY = "jlc.roofline.mem_gbps";
    private static final String GLOBAL_BANDWIDTH_ENV = "JLC_ROOFLINE_MEM_GBPS";
    private static final String L1_BW_PROPERTY = "jlc.roofline.mem_l1_gbps";
    private static final String L2_BW_PROPERTY = "jlc.roofline.mem_l2_gbps";
    private static final String L3_BW_PROPERTY = "jlc.roofline.mem_l3_gbps";
    private static final String DRAM_BW_PROPERTY = "jlc.roofline.mem_dram_gbps";
    private static final String LOOPS_PROPERTY = "jlc.roofline.mem_loops";
    private static final String RUNS_PROPERTY = "jlc.roofline.mem_runs";

    private static final int L1_SIZE_BYTES = 32 * 1024;
    private static final int L2_SIZE_BYTES = 256 * 1024;
    private static final int L3_SIZE_BYTES = 8 * 1024 * 1024;

    private static final int DEFAULT_LOOPS = 4;
    private static final int DEFAULT_RUNS = 6;
    private static final int DEFAULT_WARMUP = 2;
    private static final int L1_ELEMENTS = 1_024;        // ~24 KiB read working set.
    private static final int L2_ELEMENTS = 8_192;        // ~192 KiB read working set.
    private static final int L3_ELEMENTS = 262_144;      // ~6 MiB read working set.
    private static final int DRAM_ELEMENTS = 4_194_304;  // ~96 MiB read working set.

    private static volatile double sink;

    private MemoryBandwidthProbe() {
    }

    static BandwidthHierarchy probe() {
        Double explicit = parsePositiveDouble(System.getProperty(GLOBAL_BANDWIDTH_PROPERTY));
        if (explicit != null) {
            return BandwidthHierarchy.flat(explicit * 1e9, "property:" + GLOBAL_BANDWIDTH_PROPERTY);
        }
        explicit = parsePositiveDouble(System.getenv(GLOBAL_BANDWIDTH_ENV));
        if (explicit != null) {
            return BandwidthHierarchy.flat(explicit * 1e9, "env:" + GLOBAL_BANDWIDTH_ENV);
        }

        int loops = parsePositiveInt(System.getProperty(LOOPS_PROPERTY), DEFAULT_LOOPS);
        int runs = parsePositiveInt(System.getProperty(RUNS_PROPERTY), DEFAULT_RUNS);

        double l1 = parseOrMeasure(L1_BW_PROPERTY, L1_ELEMENTS, loops, runs);
        double l2 = parseOrMeasure(L2_BW_PROPERTY, L2_ELEMENTS, loops, runs);
        double l3 = parseOrMeasure(L3_BW_PROPERTY, L3_ELEMENTS, loops, runs);
        double dram = parseOrMeasure(DRAM_BW_PROPERTY, DRAM_ELEMENTS, loops, runs);

        return new BandwidthHierarchy(
            l1,
            l2,
            l3,
            dram,
            L1_SIZE_BYTES,
            L2_SIZE_BYTES,
            L3_SIZE_BYTES,
            "measured-hierarchy-triad"
        );
    }

    private static double parseOrMeasure(String propertyName, int elements, int loops, int runs) {
        Double explicit = parsePositiveDouble(System.getProperty(propertyName));
        if (explicit != null) {
            return explicit * 1e9;
        }
        return measureForElements(elements, loops, runs);
    }

    private static double measureForElements(int elements, int loops, int runs) {
        // Keep a roughly comparable amount of work per level to avoid tiny-loop timer bias.
        int scaledLoops = Math.max(loops, 1_048_576 / Math.max(1, elements));

        double[] a = new double[elements];
        double[] b = new double[elements];
        double[] c = new double[elements];
        for (int i = 0; i < elements; i++) {
            b[i] = 1.0 + (i % 7) * 0.1;
            c[i] = 2.0 + (i % 5) * 0.2;
        }

        for (int i = 0; i < DEFAULT_WARMUP; i++) {
            runTriad(a, b, c, scaledLoops);
        }

        double best = 0.0;
        for (int i = 0; i < runs; i++) {
            best = Math.max(best, runTriad(a, b, c, scaledLoops));
        }
        return best;
    }

    private static double runTriad(double[] a, double[] b, double[] c, int loops) {
        final int n = a.length;
        final double scalar = 1.00000011921;
        long start = System.nanoTime();
        for (int l = 0; l < loops; l++) {
            for (int i = 0; i < n; i++) {
                a[i] = b[i] + scalar * c[i];
            }
            double[] tmp = b;
            b = a;
            a = c;
            c = tmp;
        }
        long end = System.nanoTime();

        sink += b[(loops * 131) % n];

        double seconds = Math.max(1e-9, (end - start) / 1e9);
        double bytesMoved = (double) loops * n * (3.0 * Double.BYTES + Double.BYTES);
        return bytesMoved / seconds;
    }

    private static int parsePositiveInt(String value, int defaultValue) {
        if (value == null) {
            return defaultValue;
        }
        try {
            int parsed = Integer.parseInt(value.trim());
            return parsed > 0 ? parsed : defaultValue;
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static Double parsePositiveDouble(String value) {
        if (value == null) {
            return null;
        }
        try {
            double parsed = Double.parseDouble(value.trim());
            return parsed > 0.0 ? parsed : null;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    enum MemoryLevel {
        L1, L2, L3, DRAM
    }

    static final class BandwidthSelection {
        final MemoryLevel level;
        final double bytesPerSecond;

        BandwidthSelection(MemoryLevel level, double bytesPerSecond) {
            this.level = level;
            this.bytesPerSecond = bytesPerSecond;
        }
    }

    static final class BandwidthHierarchy {
        final double l1BytesPerSecond;
        final double l2BytesPerSecond;
        final double l3BytesPerSecond;
        final double dramBytesPerSecond;
        final int l1SizeBytes;
        final int l2SizeBytes;
        final int l3SizeBytes;
        final String source;

        BandwidthHierarchy(double l1BytesPerSecond,
                           double l2BytesPerSecond,
                           double l3BytesPerSecond,
                           double dramBytesPerSecond,
                           int l1SizeBytes,
                           int l2SizeBytes,
                           int l3SizeBytes,
                           String source) {
            this.l1BytesPerSecond = l1BytesPerSecond;
            this.l2BytesPerSecond = l2BytesPerSecond;
            this.l3BytesPerSecond = l3BytesPerSecond;
            this.dramBytesPerSecond = dramBytesPerSecond;
            this.l1SizeBytes = l1SizeBytes;
            this.l2SizeBytes = l2SizeBytes;
            this.l3SizeBytes = l3SizeBytes;
            this.source = source;
        }

        static BandwidthHierarchy flat(double bytesPerSecond, String source) {
            return new BandwidthHierarchy(
                bytesPerSecond,
                bytesPerSecond,
                bytesPerSecond,
                bytesPerSecond,
                L1_SIZE_BYTES,
                L2_SIZE_BYTES,
                L3_SIZE_BYTES,
                source
            );
        }

        BandwidthSelection forWorkingSet(double workingSetBytes) {
            if (workingSetBytes <= l1SizeBytes) {
                return new BandwidthSelection(MemoryLevel.L1, l1BytesPerSecond);
            }
            if (workingSetBytes <= l2SizeBytes) {
                return new BandwidthSelection(MemoryLevel.L2, l2BytesPerSecond);
            }
            if (workingSetBytes <= l3SizeBytes) {
                return new BandwidthSelection(MemoryLevel.L3, l3BytesPerSecond);
            }
            return new BandwidthSelection(MemoryLevel.DRAM, dramBytesPerSecond);
        }
    }
}
