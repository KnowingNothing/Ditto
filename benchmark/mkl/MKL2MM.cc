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
using namespace std;
int batch = 1;
int M = 512;
int N = 64;
int K = 64;
int L = 512;
double peakGflops;
#define flushsz (1 << 20) 
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
    if (argc == 7)
        repeat = atoi(argv[6]);
    if (argc == 8)
        peakGflops = atof(argv[7]);
    float *A_data = (float *)malloc(batch * M * K * sizeof(float));
    for (int j = 0; j < batch * M * K; j++)
        A_data[j] = static_cast<float>((rand() % 999) / 999.0);
    float *B_data = (float *)malloc(batch * K * L * sizeof(float));
    for (int j = 0; j < batch * K * L; j++)
        B_data[j] = static_cast<float>((rand() % 999) / 999.0);
    float *D_data = (float *)malloc(batch * L * N * sizeof(float));
    for (int j = 0; j < batch * L * N; j++)
        D_data[j] = static_cast<float>((rand() % 999) / 999.0);
    const float *A[batch];
    const float *B[batch];
    float *C[batch];
    const float *__C[batch];
    const float *D[batch];
    float *E[batch];

    const int __M[1] = {M};
    const int __N[1] = {N};
    const int __K[1] = {K};
    const int __L[1] = {L};
    const int __B[1] = {batch};
    const float __alpha[1] = {1.0f};
    const float __beta[1] = {0.0f};
    CBLAS_TRANSPOSE __trans[1] = {CblasNoTrans};
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

    printf("B, M, N, K, L, peakflops: %d %d %d %d %d %f\n", batch, M, N, K, L, peakGflops);
    float alpha = 1.0f;
    float beta = 0.0f;
    // warm up
    std::chrono::high_resolution_clock::time_point t1, t1_, t2_;

    double mkl_time = 0;
    double mkl_time_1 = 0;
    double cpu_time = 0;
    double wall_time = 0;
    for (size_t _ = 0; _ < repeat; _++)
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
            A,
            __K,
            B,
            __L,
            __beta,
            C,
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
            __C,
            __L,
            D,
            __N,
            __beta,
            E,
            __N,
            1,
            __B);
        t2_ = std::chrono::high_resolution_clock::now();
        mkl_time_1 += (double)(std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_)).count();
        for (int j = 0; j < flushsz; j ++)  
            dirty[j] = 1 / (j+1);
    }
    mkl_time /= repeat;
    mkl_time_1 /= repeat;
    cpu_time /= repeat;
    wall_time /= repeat;
    // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
    float wl = batch * (M * K * L + M * N * L);
    float ratioToPeak = wl / mkl_time_1 / 1e9 / peakGflops;
    float *groundTruthC = (float *)malloc(batch * M * L * sizeof(float));
    float *groundTruthE = (float *)malloc(batch * M * N * sizeof(float));

    groundTruth(A, B, groundTruthC, M, K, L);
    groundTruth(__C, D, groundTruthE, M, L, N);

    double maxRtol = 0, maxAtol = 0;
    for (int i = 0; i < batch; i++)
    {
        double errorCnt = 0;
        for (int j = 0; j < M * N; j++)
        {
            double tmp = fabs((E[i][j] - groundTruthE[i * M * N + j]) / E[i][j]);
            double atmp = fabs(E[i][j] - groundTruthE[i * M * N + j]);
            maxRtol = std::max(maxRtol, tmp);
            maxAtol = std::max(maxAtol, atmp);
        }
    }
    std::cout << "rtol: " << maxRtol << ", atol: " << maxAtol << std::endl;
    std::cout << "time: " << mkl_time_1 << std::endl;
    std::cout << "ratioToPeak: " << ratioToPeak << std::endl
              << std::flush;
    int n_thread = omp_get_max_threads();
    printf("n_thread: %d\n", n_thread);
    return 0;
}
