package net.faulj.benchmark;

import net.faulj.decomposition.hessenberg.BlockedHessenberg;
import net.faulj.decomposition.hessenberg.HessenbergReduction;
import net.faulj.decomposition.qr.HouseholderQR;
import net.faulj.matrix.Matrix;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

public class BenchmarkQRHess {

    private static final String MAPPING =
        "2,2\n" +
        "3,3\n" +
        "4,2\n" +
        "5,3\n" +
        "6,6\n" +
        "7,7\n" +
        "8,4\n" +
        "9,4\n" +
        "10,4\n" +
        "11,7\n" +
        "12,4\n" +
        "13,7\n" +
        "14,14\n" +
        "15,9\n" +
        "16,16\n" +
        "17,8\n" +
        "18,10\n" +
        "19,11\n" +
        "20,8\n" +
        "21,16\n" +
        "22,16\n" +
        "23,12\n" +
        "24,12\n" +
        "25,12\n" +
        "26,16\n" +
        "27,15\n" +
        "28,14\n" +
        "29,15\n" +
        "30,16\n" +
        "31,20\n" +
        "32,16\n" +
        "33,17\n" +
        "34,16\n" +
        "35,20\n" +
        "36,18\n" +
        "37,19\n" +
        "38,20\n" +
        "39,20\n" +
        "40,20\n" +
        "41,16\n" +
        "42,16\n" +
        "43,23\n" +
        "44,16\n" +
        "45,24\n" +
        "46,16\n" +
        "47,16\n" +
        "48,16\n" +
        "49,16\n" +
        "50,20\n" +
        "51,20\n" +
        "52,20\n" +
        "53,20\n" +
        "54,20\n" +
        "55,20\n" +
        "56,16\n" +
        "57,20\n" +
        "58,20\n" +
        "59,20\n" +
        "60,20\n" +
        "61,16\n" +
        "62,16\n" +
        "63,23\n" +
        "64,23\n" +
        "65,24\n" +
        "66,24\n" +
        "67,24\n" +
        "68,23\n" +
        "69,24\n" +
        "70,20\n" +
        "71,24\n" +
        "72,24\n" +
        "73,20\n" +
        "74,20\n" +
        "75,28\n" +
        "76,20\n" +
        "77,20\n" +
        "78,20\n" +
        "79,24\n" +
        "80,20\n" +
        "81,24\n" +
        "82,28\n" +
        "83,28\n" +
        "84,24\n" +
        "85,20\n" +
        "86,24\n" +
        "87,32\n" +
        "88,24\n" +
        "89,20\n" +
        "90,24\n" +
        "91,24\n" +
        "92,20\n" +
        "93,20\n" +
        "94,24\n" +
        "95,20\n" +
        "96,20\n" +
        "97,20\n" +
        "98,20\n" +
        "99,28\n" +
        "100,20\n" +
        "101,20\n" +
        "102,28\n" +
        "103,24\n" +
        "104,28\n" +
        "105,24\n" +
        "106,28\n" +
        "107,27\n" +
        "108,28\n" +
        "109,24\n" +
        "110,24\n" +
        "111,28\n" +
        "112,28\n" +
        "113,24\n" +
        "114,24\n" +
        "115,24\n" +
        "116,24\n" +
        "117,24\n" +
        "118,24\n" +
        "119,24\n" +
        "120,32\n" +
        "121,24\n" +
        "122,28\n" +
        "123,28\n" +
        "124,32\n" +
        "125,28\n" +
        "126,28\n" +
        "127,28\n" +
        "128,32\n" +
        "129,24\n" +
        "130,28\n" +
        "131,24\n" +
        "132,28\n" +
        "133,28\n" +
        "134,24\n" +
        "135,28\n" +
        "136,28\n" +
        "137,28\n" +
        "138,28\n" +
        "139,24\n" +
        "140,28\n" +
        "141,24\n" +
        "142,24\n" +
        "143,24\n" +
        "144,24\n" +
        "145,32\n" +
        "146,28\n" +
        "147,27\n" +
        "148,32\n" +
        "149,28\n" +
        "150,28\n" +
        "151,32\n" +
        "152,24\n" +
        "153,32\n" +
        "154,24\n" +
        "155,24\n" +
        "156,28\n" +
        "157,28\n" +
        "158,28\n" +
        "159,28\n" +
        "160,32\n" +
        "161,32\n" +
        "162,28\n" +
        "163,28\n" +
        "164,24\n" +
        "165,28\n" +
        "166,24\n" +
        "167,28\n" +
        "168,24\n" +
        "169,28\n" +
        "170,36\n" +
        "171,36\n" +
        "172,36\n" +
        "173,32\n" +
        "174,32\n" +
        "175,28\n" +
        "176,36\n" +
        "177,28\n" +
        "178,24\n" +
        "179,32\n" +
        "180,24\n" +
        "181,28\n" +
        "182,28\n" +
        "183,32\n" +
        "184,32\n" +
        "185,32\n" +
        "186,28\n" +
        "187,24\n" +
        "188,32\n" +
        "189,24\n" +
        "190,28\n" +
        "191,28\n" +
        "192,40\n" +
        "193,28\n" +
        "194,28\n" +
        "195,28\n" +
        "196,28\n" +
        "197,40\n" +
        "198,36\n" +
        "199,40\n" +
        "200,28\n" +
        "201,24\n" +
        "202,28\n" +
        "203,24\n" +
        "204,32\n" +
        "205,36\n" +
        "206,28\n" +
        "207,32\n" +
        "208,28\n" +
        "209,28\n" +
        "210,36\n" +
        "211,32\n" +
        "212,36\n" +
        "213,28\n" +
        "214,24\n" +
        "215,28\n" +
        "216,32\n" +
        "217,44\n" +
        "218,28\n" +
        "219,28\n" +
        "220,28\n" +
        "221,32\n" +
        "222,32\n" +
        "223,40\n" +
        "224,28\n" +
        "225,28\n" +
        "226,24\n" +
        "227,28\n" +
        "228,24\n" +
        "229,32\n" +
        "230,24\n" +
        "231,36\n" +
        "232,36\n" +
        "233,36\n" +
        "234,24\n" +
        "235,40\n" +
        "236,28\n" +
        "237,36\n" +
        "238,32\n" +
        "239,36\n" +
        "240,28\n" +
        "241,32\n" +
        "242,28\n" +
        "243,28\n" +
        "244,28\n" +
        "245,36\n" +
        "246,36\n" +
        "247,36\n" +
        "248,36\n" +
        "249,28\n" +
        "250,32\n" +
        "251,28\n" +
        "252,36\n" +
        "253,36\n" +
        "254,32\n" +
        "255,40\n" +
        "256,28\n";

    private static Map<Integer,Integer> parseMapping() {
        Map<Integer,Integer> map = new HashMap<>();
        String[] lines = MAPPING.split("\\n");
        for (String L : lines) {
            if (L.trim().isEmpty()) continue;
            String[] parts = L.split(",");
            try {
                int n = Integer.parseInt(parts[0].trim());
                int bs = Integer.parseInt(parts[1].trim());
                map.put(n, bs);
            } catch (Exception ex) {
                // ignore
            }
        }
        return map;
    }

    public static void main(String[] args) throws Exception {
        Map<Integer,Integer> map = parseMapping();
        PrintWriter out = new PrintWriter(new FileWriter("benchmark_qr_hess.csv"));
        out.println("n,algorithm,blockSize,timeNs,flops,flopsPerSec,gflops");

        for (int n = 2; n <= 256; n++) {
            int bs = map.getOrDefault(n, 16);

            double[] data = new double[n * n];
            Random rnd = new Random(12345 + n);
            for (int i = 0; i < data.length; i++) data[i] = rnd.nextDouble() - 0.5;

            final int size = n;
            final double[] dataVar = data;
            final Supplier<Matrix> baseSupplier = () -> Matrix.wrap(dataVar.clone(), size, size);

            // BlockedHessenberg
            BlockedHessenberg.setBlockSize(bs);
            long tBlocked = timeRoutine(() -> {
                Matrix A = baseSupplier.get();
                BlockedHessenberg.decompose(A);
            });
            double flopsBlocked = computeHessenbergFlops(n);
            double flopsPerSecBlocked = flopsBlocked * 1e9 / ((double) tBlocked);
            double gflopsBlocked = flopsPerSecBlocked / 1e9;
            out.printf("%d,BlockedHessenberg,%d,%d,%.0f,%.0f,%.3f\n", n, bs, tBlocked, flopsBlocked, flopsPerSecBlocked, gflopsBlocked);

            // HessenbergReduction
            long tHess = timeRoutine(() -> {
                Matrix A = baseSupplier.get();
                HessenbergReduction.decompose(A);
            });
            double flopsHess = computeHessenbergFlops(n);
            double flopsPerSecHess = flopsHess * 1e9 / ((double) tHess);
            double gflopsHess = flopsPerSecHess / 1e9;
            out.printf("%d,HessenbergReduction,,%d,%.0f,%.0f,%.3f\n", n, tHess, flopsHess, flopsPerSecHess, gflopsHess);

            // Householder QR
            long tQR = timeRoutine(() -> {
                Matrix A = baseSupplier.get();
                HouseholderQR.decompose(A);
            });
            double flopsQR = computeQRFlops(n);
            double flopsPerSecQR = flopsQR * 1e9 / ((double) tQR);
            double gflopsQR = flopsPerSecQR / 1e9;
            out.printf("%d,HouseholderQR,,%d,%.0f,%.0f,%.3f\n", n, tQR, flopsQR, flopsPerSecQR, gflopsQR);

            out.flush();
            System.out.printf("n=%d done: blocked=%d ns, hess=%d ns, qr=%d ns\n", n, tBlocked, tHess, tQR);
        }

        out.close();
        System.out.println("Benchmark finished. Results -> benchmark_qr_hess.csv");
    }

    private static long timeRoutine(Runnable r) throws InterruptedException {
        for (int i = 0; i < 3; i++) r.run();
        System.gc();
        Thread.sleep(50);

        int runs = 5;
        long best = Long.MAX_VALUE;
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            r.run();
            long dt = System.nanoTime() - t0;
            if (dt < best) best = dt;
            System.gc();
        }
        return best;
    }

    private static double computeQRFlops(int n) {
        // Householder QR for n x n: approx 2/3 * n^3 flops
        return (2.0 / 3.0) * n * (double) n * (double) n;
    }

    private static double computeHessenbergFlops(int n) {
        // Hessenberg reduction via Householder: approx 10/3 * n^3 flops
        return (10.0 / 3.0) * n * (double) n * (double) n;
    }
}
