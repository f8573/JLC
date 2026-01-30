#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

static void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        cerr << "CUDA error " << msg << ": " << cudaGetErrorString(e) << endl;
        exit(1);
    }
}

static void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        cerr << "cuBLAS error " << msg << ": " << s << endl;
        exit(1);
    }
}

template<typename T>
void run_gemm_cublas(int N, int iterations, const string &precision) {
    size_t bytes = size_t(N) * size_t(N) * sizeof(T);
    vector<T> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) { hA[i] = static_cast<T>(1.0); hB[i] = static_cast<T>(1.0); }

    T *dA=nullptr, *dB=nullptr, *dC=nullptr;
    checkCuda(cudaMalloc(&dA, bytes), "malloc A");
    checkCuda(cudaMalloc(&dB, bytes), "malloc B");
    checkCuda(cudaMalloc(&dC, bytes), "malloc C");

    checkCuda(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    float elapsed_ms = 0.0f;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    if constexpr (is_same<T,float>::value) {
        const float alpha = 1.0f, beta = 0.0f;
        checkCuda(cudaEventRecord(start), "record start");
        for (int it = 0; it < iterations; ++it) {
            checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, dA, N, dB, N, &beta, dC, N), "sgemm");
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    } else {
        const double alpha = 1.0, beta = 0.0;
        checkCuda(cudaEventRecord(start), "record start");
        for (int it = 0; it < iterations; ++it) {
            checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, dA, N, dB, N, &beta, dC, N), "dgemm");
        }
        checkCuda(cudaEventRecord(stop), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");
        checkCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    }

    double sec = double(elapsed_ms) / 1000.0;
    double flops = 2.0 * double(N) * double(N) * double(N) * double(iterations);
    double gflops = flops / sec / 1e9;

    ostringstream oss;
    oss << "{";
    oss << "\"precision\":\"" << precision << "\",";
    oss << "\"N\":" << N << ",";
    oss << "\"iterations\":" << iterations << ",";
    oss << "\"time_s\":" << sec << ",";
    oss << "\"gflops\":" << gflops;
    oss << "}" << endl;
    cout << oss.str();

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    string precision = "float";
    int N = 1024;
    int iterations = 3;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "-p" || a == "--precision") { precision = argv[++i]; }
        else if (a == "-N" || a == "--size") { N = atoi(argv[++i]); }
        else if (a == "-i" || a == "--iterations") { iterations = atoi(argv[++i]); }
    }

    if (precision == "float") run_gemm_cublas<float>(N, iterations, precision);
    else run_gemm_cublas<double>(N, iterations, precision);
    return 0;
}
