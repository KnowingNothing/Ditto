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
#include <vector>
using namespace std;
int batch = 1;
int M = 512;
int N = 64;
int K = 64;
int L = 512;
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define cacheSize (32 * 1024 / 4)
#define ROUNDUP(a, b) (((a) + (b)-1) / (b) * (b))
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
void help()
{
    printf("./script B, M, N, K, L, Repeat, Parallel\n");
}
int main(int argc, char **argv)
{
    int repeat = 100;
    if (argc != 6 && argc!= 7)
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
    
    float * A_data = (float *)_mm_malloc(batch * M * K * sizeof(float), 64);
    for (int j = 0; j < batch * M * K; j++)
        A_data[j] = static_cast<float>((rand() % 999) / 999.0);
    float * B_data = (float *)_mm_malloc(batch * K * L * sizeof(float), 64);
    for (int j = 0; j < batch * K * L; j++)
        B_data[j] = static_cast<float>((rand() % 999) / 999.0);
    float * D_data = (float *)_mm_malloc(batch * L * N * sizeof(float), 64);
    for (int j = 0; j < batch * L * N; j++)
        D_data[j] = static_cast<float>((rand() % 999) / 999.0);
    float * C_data = (float *)_mm_malloc(batch * K * L * sizeof(float), 64);
    const float *A[batch]; 
    const float *B[batch]; 
    float *C[batch]; 
    const float * __C[batch];
    const float *D[batch]; 
    float *E[batch]; 

    const int __M[1] = {M};
    const int __N[1] = {N};
    const int __K[1] = {K};
    const int __L[1] = {L};
    const int __B[1] = {batch};
    const float __alpha[1] = {1.0f};
    const float __beta[1] = {0.0f};
    CBLAS_TRANSPOSE    __trans[1] = {CblasNoTrans};
    for (size_t i = 0; i < batch; i++){
        A[i] = A_data + M * K * i;
        B[i] = B_data + K * L * i;
        C[i] = (float *)_mm_malloc(M * L * sizeof(float), 64);
        __C[i] = (float *)_mm_malloc(M * L * sizeof(float), 64);
        D[i] = D_data + L * N * i;
        E[i] = (float *)_mm_malloc(M * N * sizeof(float), 64);
    }

    printf("M, N, K: %d %d %d\n", M, N, K);
    float alpha = 1.0f;
    float beta = 0.0f;
    // warm up
    std::chrono::high_resolution_clock::time_point t1, t1_, t2_;
    
    // t1 = std::chrono::high_resolution_clock::now();
    double mkl_time = 0;
    double mkl_time_1 = 0;
    double cpu_time = 0;
    double wall_time = 0;
    for (size_t _ = 0; _ < repeat; _++){
        // std::clock_t c_start = std::clock();
        t1_ = std::chrono::high_resolution_clock::now();
        // double cpu_time_start = get_cpu_time();
        // double wall_time_start = get_wall_time();
        cblas_sgemm_batch (
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
        cblas_sgemm_batch (
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
        // double cpu_time_end = get_cpu_time();
        // double wall_time_end = get_wall_time();
        // t1 = std::chrono::high_resolution_clock::now();
        t2_ = std::chrono::high_resolution_clock::now();
        // std::clock_t c_end = std::clock();
        // double time_elapsed_s = (c_end-c_start) / (double)CLOCKS_PER_SEC;
        mkl_time_1 += (double)(std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_)).count();
        // mkl_time += time_elapsed_s;
        // cpu_time += cpu_time_end - cpu_time_start;
        // wall_time += wall_time_end - wall_time_start;
        
    }
    mkl_time /= repeat;
    mkl_time_1 /= repeat;
    cpu_time /= repeat;
    wall_time /= repeat;
    // std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
    float wl = batch * (M * K * L + M * N * L);
    float ratioToPeak = wl / mkl_time_1 / 1e9 / 2.2 / 35.2 / 20;
    std::cout << "ratioToPeak: " << ratioToPeak << std::endl
              << std::flush;
    return 0;
}
