#include <iostream>
#include <string>
#include <chrono>
#include <assert.h>
#include <stdio.h>


using namespace std;

void sgemm(float * A, float * B, float * C, int M, int N, int K, int Tm, int Tn, int Tk){
    for (int io = 0; io < M; io += Tm){
        for (int jo = 0; jo < N; jo += Tn){
            for (int ko = 0; ko < K; ko += Tk){
                for (int ii = 0; ii < Tm && ii < (M - io); ii ++){
                    for (int ji = 0; ji < Tn && ji < (N - jo); ji ++){
                        for (int ki = 0; ki < Tk && ki < (K - ko); ki ++){
                            int i = io + ii;
                            int j = jo + ji;
                            int k = ko + ki;
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
    return;
}


int main(int argc, char ** argv){
    int M, N, K, Tm, Tn, Tk;
    int repeat = 1;

    assert(argc >= 7);
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    Tm = atoi(argv[4]);
    Tn = atoi(argv[5]);
    Tk = atoi(argv[6]);
    if (argc >= 8) repeat = atoi(argv[7]);
    

    float * A, *B, *C;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(N * K * sizeof(float));
    C = (float*)malloc(M * N * sizeof(float));
#ifdef PROFILE
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    printf("GEMM M,N,K,Tm,Tn,Tk: %d, %d, %d, %d, %d, %d, \n", M, N, K, Tm, Tn, Tk);
#endif 
    for (int i = 0; i < repeat; i++){
        sgemm(A, B, C, M, N, K, Tm, Tn, Tk);
    }
#ifdef PROFILE
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " µs" << std::endl;
#endif 
    return 0;
}

