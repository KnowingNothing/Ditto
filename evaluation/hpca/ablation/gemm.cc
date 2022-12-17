
    #include <mkl.h>
    #include <unistd.h>
    #include <stdio.h>
    #include <immintrin.h>
    extern "C" int gemm2_update_baseline(float * C, float *A, float*B, int K_, int N_, int c_stride_){
        for (int k = 0; k < 64; k++){
            for (int i = 0; i < 6; i++){
                for (int j = 0; j < 64; j++){
                    C[i * c_stride_ + j] += A[i * K_ + k] * B[k * N_ + j];
                }
            }
        }
        return 0;
    }
    extern "C" int gemm2_update_mkl(float * C, float *A, float*B, int K_, int N_, int c_stride_){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                    6, 64, 64, 1.0, 
                                    A, K_, 
                                    B, N_, 1.0, 
                                    C, c_stride_);
        return 0;
    }
    
extern "C" int gemm2_update(float *C, float *A, float *B, int K_, int N_, int c_stride_)
{
    getpid();
    long long K = K_;
    long long N = N_;
    long long c_stride = c_stride_;

    __asm__ __volatile__(
        // AT&T syntax: src dst
        "mov %[A], %%rax;"
        "mov %[B], %%rbx;"
        "mov %[C], %%rcx;"
        "mov %[K], %%rsi;"
        "mov %[N], %%rdi;"

        "mov %[c_stride], %%rdx;"
        "xor %%r8, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm0;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm1;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm2;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm3;"
        "add %%rdx, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm4;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm5;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm6;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm7;"
        "add %%rdx, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm8;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm9;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm10;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm11;"
        "add %%rdx, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm12;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm13;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm14;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm15;"
        "add %%rdx, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm16;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm17;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm18;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm19;"
        "add %%rdx, %%r8;"
        "vmovups 0(%%rcx, %%r8, 4), %%zmm20;"
        "vmovups 64(%%rcx, %%r8, 4), %%zmm21;"
        "vmovups 128(%%rcx, %%r8, 4), %%zmm22;"
        "vmovups 192(%%rcx, %%r8, 4), %%zmm23;"

        "mov %[K], %%rdx;"
        "mov $0, %%r8;"
        "lea (%%r8, %%rdx), %%r9;"
        "lea (%%r9, %%rdx), %%r10;"
        "lea (%%r10, %%rdx), %%r11;"
        "lea (%%rax, %%r11, 4), %%r11;"

        "mov $64, %%rdx;"
        "test %%rdx, %%rdx;"
        "jz .store%=;"

        ".compute%=:"
        "vmovups 0(%%rbx), %%zmm28;"
        "vmovups 64(%%rbx), %%zmm29;"
        "vmovups 128(%%rbx), %%zmm30;"
        "vmovups 192(%%rbx), %%zmm31;"
        "vbroadcastss 0(%%rax, %%r8, 4), %%zmm25;"
        "vbroadcastss 0(%%rax, %%r9, 4), %%zmm26;"
        "vbroadcastss 0(%%rax, %%r10, 4), %%zmm27;"
        "vfmadd231ps %%zmm25, %%zmm28, %%zmm0;"
        "vfmadd231ps %%zmm25, %%zmm29, %%zmm1;"
        "vfmadd231ps %%zmm25, %%zmm30, %%zmm2;"
        "vfmadd231ps %%zmm25, %%zmm31, %%zmm3;"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm4;"
        "vfmadd231ps %%zmm26, %%zmm29, %%zmm5;"
        "vfmadd231ps %%zmm26, %%zmm30, %%zmm6;"
        "vfmadd231ps %%zmm26, %%zmm31, %%zmm7;"
        "vfmadd231ps %%zmm27, %%zmm28, %%zmm8;"
        "vfmadd231ps %%zmm27, %%zmm29, %%zmm9;"
        "vfmadd231ps %%zmm27, %%zmm30, %%zmm10;"
        "vfmadd231ps %%zmm27, %%zmm31, %%zmm11;"

        "vbroadcastss 0(%%r11, %%r8, 4), %%zmm25;"
        "vbroadcastss 0(%%r11, %%r9, 4), %%zmm26;"
        "vbroadcastss 0(%%r11, %%r10, 4), %%zmm27;"
        "vfmadd231ps %%zmm25, %%zmm28, %%zmm12;"
        "vfmadd231ps %%zmm25, %%zmm29, %%zmm13;"
        "vfmadd231ps %%zmm25, %%zmm30, %%zmm14;"
        "vfmadd231ps %%zmm25, %%zmm31, %%zmm15;"
        "vfmadd231ps %%zmm26, %%zmm28, %%zmm16;"
        "vfmadd231ps %%zmm26, %%zmm29, %%zmm17;"
        "vfmadd231ps %%zmm26, %%zmm30, %%zmm18;"
        "vfmadd231ps %%zmm26, %%zmm31, %%zmm19;"
        "vfmadd231ps %%zmm27, %%zmm28, %%zmm20;"
        "vfmadd231ps %%zmm27, %%zmm29, %%zmm21;"
        "vfmadd231ps %%zmm27, %%zmm30, %%zmm22;"
        "vfmadd231ps %%zmm27, %%zmm31, %%zmm23;"

        "lea 4(%%rax), %%rax;"
        "lea 4(%%r11), %%r11;"
        "lea 0(%%rbx, %%rdi, 4), %%rbx;"
        "sub $1, %%rdx;"
        "jnz .compute%=;"

        ".store%=:"
        "mov %[c_stride], %%rdx;"
        "xor %%r8, %%r8;"
        "vmovups %%zmm0, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm1, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm2, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm3, 192(%%rcx, %%r8, 4);"
        "add %%rdx, %%r8;"
        "vmovups %%zmm4, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm5, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm6, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm7, 192(%%rcx, %%r8, 4);"
        "add %%rdx, %%r8;"
        "vmovups %%zmm8, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm9, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm10, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm11, 192(%%rcx, %%r8, 4);"
        "add %%rdx, %%r8;"
        "vmovups %%zmm12, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm13, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm14, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm15, 192(%%rcx, %%r8, 4);"
        "add %%rdx, %%r8;"
        "vmovups %%zmm16, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm17, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm18, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm19, 192(%%rcx, %%r8, 4);"
        "add %%rdx, %%r8;"
        "vmovups %%zmm20, 0(%%rcx, %%r8, 4);"
        "vmovups %%zmm21, 64(%%rcx, %%r8, 4);"
        "vmovups %%zmm22, 128(%%rcx, %%r8, 4);"
        "vmovups %%zmm23, 192(%%rcx, %%r8, 4);"
        :
        : [A] "m"(A),
          [B] "m"(B),
          [C] "m"(C),
          [K] "m"(K),
          [N] "m"(N),
          [c_stride] "m"(c_stride)
        : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", 
        "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31");
    getpid();
    return 0;
}
extern "C" int gemm2_reset(float *cc, int stride) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) 
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            cc[i*stride+j] = 0.0;
        }
    return 0;
}
        