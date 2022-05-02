#include <immintrin.h>
#include <time.h>
#include <cstring>
#include <stdio.h>
#include <sys/time.h>

#include <math.h>
#include <cstdlib>
#include <time.h>
#include <immintrin.h>
#include <iostream>
#include <type_traits>
#include <string>
#include <chrono>
#include <ctime>
#include <mkl.h>
#include <omp.h>
#include <vector>
#include <thread>
#include <emmintrin.h>
#include <likwid.h>
#include <likwid-marker.h>
using namespace std;
int batch = 1;
int M = 512;
int N = 64;
int K = 64;
int L = 512;
double peakGflops;

#define flushsz (1 << 26)
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define cacheSize (32 * 1024 / 4)
#define ROUNDUP(a, b) (((a) + (b)-1) / (b) * (b))
double dirty[flushsz];
double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time()
{
    return (double)clock() / CLOCKS_PER_SEC;
}
void help()
{
    printf("./script B, M, N, K, L, Repeat, PeakGlops\n");
    printf("sc: 704, sccc: 2252.8\n");
}

void groundTruth(const float **A, const float **B, float *C, int M, int K, int L)
{
    for (int b = 0; b < batch; b++)
    {
        const float *__A = A[b];
        const float *__B = B[b];
        float *__C = C + M * L * b;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < L; j++)
            {
                __C[i * L + j] = 0;
                for (int k = 0; k < K; k++)
                    __C[i * L + j] += __A[i * K + k] * __B[k * L + j];
            }
    }
}
int main(int argc, char **argv)
{
    int repeat = 100;
    if (argc != 6 && argc != 7 && argc != 8)
    {
        help();
        return 0;
    }
    batch = atoi(argv[1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    L = atoi(argv[5]);
    if (argc >= 7)
        repeat = atoi(argv[6]);
    if (argc == 8)
        peakGflops = atof(argv[7]);

    struct Data
    {
        float *A_data;
        float *B_data;
        float *D_data;
        const float *A[20];
        const float *B[20];
        float *C[20];
        const float *__C[20];
        const float *D[20];
        float *E[20];
        Data()
        {
            A_data = (float *)malloc(batch * M * K * sizeof(float));
            B_data = (float *)malloc(batch * K * L * sizeof(float));
            D_data = (float *)malloc(batch * L * N * sizeof(float));
            for (size_t i = 0; i < batch; i++)
            {
                A[i] = A_data + M * K * i;
                B[i] = B_data + K * L * i;
                C[i] = (float *)malloc(M * L * sizeof(float));
                __C[i] = C[i];
                D[i] = D_data + L * N * i;
                E[i] = (float *)malloc(M * N * sizeof(float));
                memset(C[i], 0, M * L * sizeof(float));
                memset(E[i], 0, M * N * sizeof(float));
            }
            for (int j = 0; j < batch * M * K; j++)
                A_data[j] = static_cast<float>((rand() % 999) / 999.0);
            for (int j = 0; j < batch * K * L; j++)
                B_data[j] = static_cast<float>((rand() % 999) / 999.0);
            for (int j = 0; j < batch * L * N; j++)
                D_data[j] = static_cast<float>((rand() % 999) / 999.0);
        }
    } data[10];

    const int __M[1] = {M};
    const int __N[1] = {N};
    const int __K[1] = {K};
    const int __L[1] = {L};
    const int __B[1] = {batch};
    const float __alpha[1] = {1.0f};
    const float __beta[1] = {0.0f};
    CBLAS_TRANSPOSE __trans[1] = {CblasNoTrans};

    printf("B, M, N, K, L, peakflops: %d %d %d %d %d %f\n", batch, M, N, K, L, peakGflops);
    float alpha = 1.0f;
    float beta = 0.0f;
    auto flush = [](const float *array, const int size) {
        for (int i = 0; i < size; i += 64 / sizeof(float))
        {
            _mm_clflush(&array[i]);
        }
    };
    std::chrono::high_resolution_clock::time_point start, end, t1_, t2_;
    double time = 0;
    // LIKWID_MARKER_INIT;

    // LIKWID_MARKER_THREADINIT;
    // for (size_t _ = 0; _ < repeat; _++)
    // {
    //     flush(A_data, batch * M * K);
    //     flush(B_data, batch * L * K);
    //     flush(D_data, batch * L * N);
    //     for (int i = 0; i < batch; i++){
    //         flush(C[i], M * L);
    //         flush(E[i], N * M);
    //     }
    //     t1_ = std::chrono::high_resolution_clock::now();
    //     LIKWID_MARKER_START("Compute");
    //     cblas_sgemm_batch(
    //         CblasRowMajor,
    //         __trans,
    //         __trans,
    //         __M,
    //         __L,
    //         __K,
    //         __alpha,
    //         A,
    //         __K,
    //         B,
    //         __L,
    //         __beta,
    //         C,
    //         __L,
    //         1,
    //         __B);
    //     cblas_sgemm_batch(
    //         CblasRowMajor,
    //         __trans,
    //         __trans,
    //         __M,
    //         __N,
    //         __L,
    //         __alpha,
    //         __C,
    //         __L,
    //         D,
    //         __N,
    //         __beta,
    //         E,
    //         __N,
    //         1,
    //         __B);
    //     t2_ = std::chrono::high_resolution_clock::now();
    //     LIKWID_MARKER_STOP("Compute");
    //     time += (double)(std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_)).count();
    // }
    // LIKWID_MARKER_CLOSE;
    // time /= repeat;
    // printf("likwid_time %f\n", time);

    // time = 0;
    for (size_t _ = 0; _ < repeat; _++)
    {
        // flush(A_data, batch * M * K);
        // flush(B_data, batch * L * K);
        // flush(D_data, batch * L * N);
        // for (int i = 0; i < batch; i++){
        //     flush(C[i], M * L);
        //     flush(E[i], N * M);
        // }
        for (int j = 0; j < 10; j++)
        {
            t1_ = std::chrono::high_resolution_clock::now();
            cblas_sgemm_batch(
                CblasRowMajor,
                __trans,
                __trans,
                __M,
                __L,
                __K,
                __alpha,
                data[j].A,
                __K,
                data[j].B,
                __L,
                __beta,
                data[j].C,
                __L,
                1,
                __B);
            cblas_sgemm_batch(
                CblasRowMajor,
                __trans,
                __trans,
                __M,
                __N,
                __L,
                __alpha,
                data[j].__C,
                __L,
                data[j].D,
                __N,
                __beta,
                data[j].E,
                __N,
                1,
                __B);
            t2_ = std::chrono::high_resolution_clock::now();
            time += (double)(std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_)).count();
        }
    }
    time /= (repeat * 10);
    double wl = 1.0 * batch * M * L * (N + K);
    double topeak = wl / 1e9 / time / peakGflops;
    printf("ratioToPeak: %f\n", topeak);
    printf("time: %f\n", time);
    return 0;
}
