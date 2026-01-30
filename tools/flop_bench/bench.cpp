#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

template<typename T>
void matmul_blocked(const T* A, const T* B, T* C, int N) {
    const int BS = 64;
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS) {
        for (int kk = 0; kk < N; kk += BS) {
            for (int jj = 0; jj < N; jj += BS) {
                int iMax = min(ii + BS, N);
                int kMax = min(kk + BS, N);
                int jMax = min(jj + BS, N);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        T a = A[i * N + k];
                        for (int j = jj; j < jMax; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
int run(int N, int iterations, int threads, const string &precision) {
    omp_set_num_threads(threads);

    size_t nn = size_t(N) * size_t(N);
    vector<T> A(nn), B(nn), C(nn);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < nn; ++i) {
        A[i] = static_cast<T>(dist(rng));
        B[i] = static_cast<T>(dist(rng));
        C[i] = static_cast<T>(0);
    }

    // Warmup
    matmul_blocked<T>(A.data(), B.data(), C.data(), N);
    fill(C.begin(), C.end(), static_cast<T>(0));

    auto t0 = chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it) {
        matmul_blocked<T>(A.data(), B.data(), C.data(), N);
        fill(C.begin(), C.end(), static_cast<T>(0));
    }
    auto t1 = chrono::high_resolution_clock::now();
    double sec = chrono::duration_cast<chrono::duration<double>>(t1 - t0).count();

    double flops = 2.0 * double(N) * double(N) * double(N) * double(iterations);
    double gflops = flops / sec / 1e9;

    // Output JSON line
    ostringstream oss;
    oss << "{"
        << "\"precision\":\"" << precision << "\",";
    oss << "\"N\":" << N << ",";
    oss << "\"threads\":" << threads << ",";
    oss << "\"iterations\":" << iterations << ",";
    oss << "\"time_s\":" << sec << ",";
    oss << "\"gflops\":" << gflops;
    oss << "}" << endl;
    cout << oss.str();
    return 0;
}

int main(int argc, char** argv) {
    string precision = "float";
    int N = 1024;
    int iterations = 3;
    int threads = omp_get_max_threads();

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-p" || a == "--precision") { precision = argv[++i]; }
        else if (a == "-N" || a == "--size") { N = atoi(argv[++i]); }
        else if (a == "-i" || a == "--iterations") { iterations = atoi(argv[++i]); }
        else if (a == "-t" || a == "--threads") { threads = atoi(argv[++i]); }
    }

    if (precision == "float") return run<float>(N, iterations, threads, precision);
    else return run<double>(N, iterations, threads, precision);
}
