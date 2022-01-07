import numpy as np
import tvm
from tvm import te

MICRO_M = 6  # fixed
MICRO_N = 32 # fixed
MICRO_K = 1  # reduce all as MIRCO_K

TILE1_M = 5
TILE1_N = 3
TILE1_K = 1

def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))

def intrin_micro_kernel_gemm1(dtype, MICRO_K):
    a = te.placeholder((MICRO_M, MICRO_K), name="a", dtype=dtype)
    b = te.placeholder((MICRO_K, MICRO_N), name="b", dtype=dtype)
    k = te.reduce_axis((0, MICRO_K), name="k")
    c = te.compute((MICRO_M, MICRO_N), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", data_alignment=64, offset_factor=1, strides=[te.var("s1"), 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", data_alignment=64, offset_factor=1, strides=[te.var("s2"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", data_alignment=64, offset_factor=1, strides=[te.var("s3"), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "gemm1_update",
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    aa.strides[0], bb.strides[0], cc.strides[0]
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "gemm_reset", cc.access_ptr("w"), cc.strides[0]))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def intrin_micro_kernel_gemm2(dtype, MICRO_K):
    a = te.placeholder((MICRO_M, MICRO_K), name="a", dtype=dtype)
    b = te.placeholder((MICRO_K, MICRO_N), name="b", dtype=dtype)
    k = te.reduce_axis((0, MICRO_K), name="k")
    c = te.compute((MICRO_M, MICRO_N), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", data_alignment=64, offset_factor=1, strides=[te.var("s1"), 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", data_alignment=64, offset_factor=1, strides=[te.var("s2"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", data_alignment=64, offset_factor=1, strides=[te.var("s3"), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "gemm2_update",
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    aa.strides[0], bb.strides[0], cc.strides[0]
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "gemm_reset", cc.access_ptr("w"), cc.strides[0]))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def intrin_micro_kernel_exp(dtype):
    a = te.placeholder((MICRO_M, MICRO_N), name="a", dtype=dtype)
    c = te.compute((MICRO_M, MICRO_N), lambda i, j: te.exp(a[i, j]), name="c")
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", data_alignment=64, offset_factor=1, strides=[te.var("s1"), 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", data_alignment=64, offset_factor=1, strides=[te.var("s3"), 1])

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]
        ib = tvm.tir.ir_builder.create()
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "compute_exp",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                cc.strides[0], aa.strides[0]
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})

def kernels_impl(MICRO_K1, MICRO_K2):
    cc_code = """
        #include <immintrin.h>
        #include <stdio.h>

        extern "C" int gemm1_update(float *C, float *A, float *B, int K_, int N_, int c_stride_) {
            long long K = K_;
            long long N = N_;
            long long c_stride = c_stride_;
            /*#pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    C[i*N_+j] = 0.0;
                }*/
            __asm__(
                // AT&T syntax: src dst 
                "mov %%[A], %%%%rax;"
                "mov %%[B], %%%%rbx;"
                "mov %%[C], %%%%rcx;"
                "mov %%[K], %%%%rsi;"
                "mov %%[N], %%%%rdi;"

                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm4;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm5;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm6;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm7;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm8;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm9;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm10;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm11;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm12;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm13;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm14;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm15;"

                "mov %%[K], %%%%rdx;"
                "mov $0, %%%%r8;"
                "lea (%%%%r8, %%%%rdx), %%%%r9;"
                "lea (%%%%r9, %%%%rdx), %%%%r10;"
                "lea (%%%%r10, %%%%rdx), %%%%r11;"
                "lea (%%%%r11, %%%%rdx), %%%%r12;"
                "lea (%%%%r12, %%%%rdx), %%%%r13;"

                "mov $%d, %%%%rdx;"
                "test %%%%rdx, %%%%rdx;"
                "jz .store%%=;"
                

            ".compute%%=:"
                "vmovaps 0(%%%%rbx), %%%%zmm2;"
                "vmovaps 64(%%%%rbx), %%%%zmm3;"
                "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm4;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm5;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm6;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm7;"
                "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r11, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm8;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm9;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm10;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm11;"
                "vbroadcastss 0(%%%%rax, %%%%r12, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r13, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm12;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm13;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm14;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm15;"

                "lea 4(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps %%%%zmm4, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm5, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm6, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm7, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm8, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm9, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm10, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm11, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm12, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm13, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm14, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm15, 64(%%%%rcx, %%%%r8, 4);"
                :
                :[A] "m" (A),
                [B] "m" (B),
                [C] "m" (C),
                [K] "m" (K),
                [N] "m" (N),
                [c_stride] "m" (c_stride)
                :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15"
            );
            return 0;
        }
        extern "C" int gemm2_update(float *C, float *A, float *B, int K_, int N_, int c_stride_) {
            long long K = K_;
            long long N = N_;
            long long c_stride = c_stride_;
            /*#pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    C[i*N_+j] = 0.0;
                }*/
            __asm__(
                // AT&T syntax: src dst 
                "mov %%[A], %%%%rax;"
                "mov %%[B], %%%%rbx;"
                "mov %%[C], %%%%rcx;"
                "mov %%[K], %%%%rsi;"
                "mov %%[N], %%%%rdi;"

                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm4;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm5;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm6;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm7;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm8;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm9;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm10;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm11;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm12;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm13;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%zmm14;"
                "vmovaps 64(%%%%rcx, %%%%r8, 4), %%%%zmm15;"

                "mov %%[K], %%%%rdx;"
                "mov $0, %%%%r8;"
                "lea (%%%%r8, %%%%rdx), %%%%r9;"
                "lea (%%%%r9, %%%%rdx), %%%%r10;"
                "lea (%%%%r10, %%%%rdx), %%%%r11;"
                "lea (%%%%r11, %%%%rdx), %%%%r12;"
                "lea (%%%%r12, %%%%rdx), %%%%r13;"

                "mov $%d, %%%%rdx;"
                "test %%%%rdx, %%%%rdx;"
                "jz .store%%=;"
                

            ".compute%%=:"
                "vmovaps 0(%%%%rbx), %%%%zmm2;"
                "vmovaps 64(%%%%rbx), %%%%zmm3;"
                "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm4;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm5;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm6;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm7;"
                "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r11, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm8;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm9;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm10;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm11;"
                "vbroadcastss 0(%%%%rax, %%%%r12, 4), %%%%zmm0;"
                "vbroadcastss 0(%%%%rax, %%%%r13, 4), %%%%zmm1;"
                "vfmadd231ps %%%%zmm2, %%%%zmm0, %%%%zmm12;"
                "vfmadd231ps %%%%zmm3, %%%%zmm0, %%%%zmm13;"
                "vfmadd231ps %%%%zmm2, %%%%zmm1, %%%%zmm14;"
                "vfmadd231ps %%%%zmm3, %%%%zmm1, %%%%zmm15;"

                "lea 4(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps %%%%zmm4, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm5, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm6, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm7, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm8, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm9, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm10, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm11, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm12, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm13, 64(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%zmm14, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%zmm15, 64(%%%%rcx, %%%%r8, 4);"
                :
                :[A] "m" (A),
                [B] "m" (B),
                [C] "m" (C),
                [K] "m" (K),
                [N] "m" (N),
                [c_stride] "m" (c_stride)
                :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15"
            );
            return 0;
        }
        extern "C" int gemm_reset(float *cc, int stride) {
            #pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    cc[i*stride+j] = 0.0;
                }
            return 0;
        }

        // consts
        const float exp_hi = 88.3762626647949f;
        const float exp_lo = -88.3762626647949f;
        const float cephes_LOG2EF = 1.44269504088896341;
        const float _one = 1.0f;
        const float _0p5 = 0.5f;
        const float cephes_exp_C1 = 0.693359375;
        const float cephes_exp_C2 = -2.12194440e-4;
        const float cephes_exp_p0 = 1.9875691500E-4;
        const float cephes_exp_p1 = 1.3981999507E-3;
        const float cephes_exp_p2 = 8.3334519073E-3;
        const float cephes_exp_p3 = 4.1665795894E-2;
        const float cephes_exp_p4 = 1.6666665459E-1;
        const float cephes_exp_p5 = 5.0000001201E-1;
        const int _0x7f = 0x7f;
        const __m256 _ps_exp_hi = _mm256_broadcast_ss(&exp_hi);
        const __m256 _ps_exp_lo = _mm256_broadcast_ss(&exp_lo);
        const __m256 _ps_cephes_LOG2EF = _mm256_broadcast_ss(&cephes_LOG2EF);
        const __m256 _ps_0p5 = _mm256_broadcast_ss(&_0p5);
        const __m256 _ps_one = _mm256_broadcast_ss(&_one);
        const __m256 _ps_cephes_exp_C1 = _mm256_broadcast_ss(&cephes_exp_C1);
        const __m256 _ps_cephes_exp_C2 = _mm256_broadcast_ss(&cephes_exp_C2);
        const __m256 _ps_cephes_exp_p0 = _mm256_broadcast_ss(&cephes_exp_p0);
        const __m256 _ps_cephes_exp_p1 = _mm256_broadcast_ss(&cephes_exp_p1);
        const __m256 _ps_cephes_exp_p2 = _mm256_broadcast_ss(&cephes_exp_p2);
        const __m256 _ps_cephes_exp_p3 = _mm256_broadcast_ss(&cephes_exp_p3);
        const __m256 _ps_cephes_exp_p4 = _mm256_broadcast_ss(&cephes_exp_p4);
        const __m256 _ps_cephes_exp_p5 = _mm256_broadcast_ss(&cephes_exp_p5);
        const __m256 _ps_0x7f = _mm256_broadcast_ss((float*)&_0x7f);
        const __m256 _ps_consts[28] = {
            _ps_exp_hi, _ps_exp_hi,
            _ps_exp_lo, _ps_exp_lo,
            _ps_cephes_LOG2EF, _ps_cephes_LOG2EF,
            _ps_0p5, _ps_0p5,
            _ps_one, _ps_one,
            _ps_cephes_exp_C1, _ps_cephes_exp_C1,
            _ps_cephes_exp_C2, _ps_cephes_exp_C2,
            _ps_cephes_exp_p0, _ps_cephes_exp_p0,
            _ps_cephes_exp_p1, _ps_cephes_exp_p1,
            _ps_cephes_exp_p2, _ps_cephes_exp_p2,
            _ps_cephes_exp_p3, _ps_cephes_exp_p3,
            _ps_cephes_exp_p4, _ps_cephes_exp_p4,
            _ps_cephes_exp_p5, _ps_cephes_exp_p5,
            _ps_0x7f, _ps_0x7f
        };
        const void* _ps_exp_hi_p = (void*)_ps_consts;
        const void* _ps_one_p = (void*)(_ps_consts+8);
        // consts end
        
        extern "C" int compute_exp(float* cc_, float *aa_, int stride1_, int stride2_) {
            long long stride1 = stride1_;
            long long stride2 = stride2_;
            // #pragma unroll
            for (int i = 0; i < %d; ++i){
                for(int j = 0; j < 2; ++j){
                    float* cc = cc_ + i * stride1_ + j * 16;
                    float* aa = aa_ + i * stride2_ + j * 16;
                    __asm__(
                    ".exp%%=:"
                        "mov %%[aa], %%%%rax;"
                        "mov %%[cc], %%%%rcx;"
                        "mov %%[_ps_exp_hi_p], %%%%rbx;"
                        "mov %%[_ps_one_p], %%%%rdx;"
                        "vmovaps (%%%%rax), %%%%zmm0;"

                        // _mm_min_ps
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vminps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_max_ps
                        // "mov %%[_ps_exp_lo_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vmaxps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_LOG2EF_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm0, %%%%zmm1, %%%%zmm2;"

                        // _mm_add_ps
                        // "mov %%[_ps_0p5_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vaddps %%%%zmm2, %%%%zmm1, %%%%zmm2;"

                        // _mm_cvttps_epi32
                        "vcvttps2dq %%%%zmm2, %%%%zmm1;"
                        
                        // _mm_cvtepi32_ps
                        "vcvtdq2ps %%%%zmm1, %%%%zmm1;"
                        
                        // sub if greater than
                        "vcmpps $14, %%%%zmm2, %%%%zmm1, %%%%k1;"
                        // "mov %%[_ps_one_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vsubps %%%%zmm3, %%%%zmm1, %%%%zmm2;"
                        // "vmovaps %%%%zmm2, %%%%zmm1{%%%%k1};"
                        // "vmovaps %%%%zmm1, %%%%zmm2;"
                        "vblendmps %%%%zmm2, %%%%zmm1, %%%%zmm2 %%{%%%%k1%%};"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_exp_C1_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        // _mm_sub_ps
                        "vsubps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_exp_C2_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        // _mm_sub_ps
                        "vsubps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // mul & add
                        // "mov %%[_ps_cephes_exp_p0_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p1_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p2_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p3_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p4_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p5_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        "vaddps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_one_p], %%%%rdx;"
                        "vmovaps (%%%%rdx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"

                        "vcvttps2dq %%%%zmm2, %%%%zmm2;"
                        // "mov %%[_ps_0x7f_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovaps (%%%%rbx), %%%%zmm0;"
                        "vpaddd %%%%zmm2, %%%%zmm0, %%%%zmm2;"
                        "vpslld $23, %%%%zmm2, %%%%zmm2;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        "vmovaps %%%%zmm1, (%%%%rcx);"
                        :
                        :[cc] "m" (cc),
                        [aa] "m" (aa),
                        [_ps_exp_hi_p] "m" (_ps_exp_hi_p),
                        [_ps_one_p] "m" (_ps_one_p)
                        :"rax", "rbx", "rcx", "rdx", "zmm0", "zmm1", "zmm2", "zmm3", "k1"
                    );
                    // printf("%%f ", cc[1]);
                }
            }
            return 0;
        }
    """ % (MICRO_M, MICRO_N, MICRO_K1, MICRO_M, MICRO_N, MICRO_K2, MICRO_M, MICRO_N, MICRO_M)
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-mavx", "-mavx512f", "-msse"])
    return ll_code

def gemm_chain(batch, M, K, L, N, dtype):
    M_pad = ((M - 1) // (MICRO_M * TILE1_M) + 1) * (MICRO_M * TILE1_M)
    L_pad = ((L - 1) // (MICRO_N * TILE1_N) + 1) * (MICRO_N * TILE1_N)
    K_pad = ((K - 1) // (MICRO_K * TILE1_K) + 1) * (MICRO_K * TILE1_K)
    N_pad = (N // (MICRO_N * TILE1_N) + 1) * (MICRO_N * TILE1_N)
    print(M_pad)
    print(L_pad)
    print(K_pad)
    print(N_pad)

    # define compute begin
    Qtensor = tvm.te.placeholder([batch, M, K], name="Q", dtype=dtype)
    Ktensor = tvm.te.placeholder([batch, K, L], name="K", dtype=dtype)
    Vtensor = tvm.te.placeholder([batch, L, N], name="V", dtype=dtype)

    Qtensor_pad = te.compute(
        (batch, M_pad, K_pad),
        lambda b, i, k: tvm.te.if_then_else(tvm.tir.all(i < M, k < K), Qtensor[b, i, k], 0),
        name = "Q_pad"
    )

    Ktensor_pad = te.compute(
        (batch, K_pad, L_pad),
        lambda b, k, l: tvm.te.if_then_else(tvm.tir.all(k < K, l < L), Ktensor[b, k, l], 0),
        name = "K_pad"
    )

    Vtensor_pad = te.compute(
        (batch, L_pad, N_pad),
        lambda b, l, j: tvm.te.if_then_else(l < L, tvm.te.if_then_else(j < N, Vtensor[b, l, j], tvm.te.if_then_else(j == N, 1, 0)), 0),
        name = "V_pad"
    )

    k = te.reduce_axis((0, K_pad), name="k")
    QK_pad = te.compute(
        (batch, M_pad, L_pad),
        lambda b, i, l: te.sum(Qtensor_pad[b, i, k] * Ktensor_pad[b, k, l], axis=k),
        name="QK_pad"
    )

    QK_exp = te.compute(
        (batch, M_pad, L_pad),
        lambda b, i, l: te.exp(QK_pad[b, i, l]),
        name = "QK_exp"
    )
    
    l = te.reduce_axis((0, L_pad), name="l")
    QKV_t = te.compute(
        (batch, M_pad, N_pad),
        lambda b, i, j: te.sum(QK_exp[b, i, l] * Vtensor_pad[b, l, j], axis=l),
        name="QKV_t"
    )

    if M_pad == M and N_pad == N+1:
        QKV = te.compute((batch, M, N), lambda b, i, j: QKV_t[b, i, j] / QKV_t[b, i, N], name = "QKV")
    else:
        QKV = te.compute((batch, M, N), lambda b, i, j: (QKV_t[b, i, j] +  QKV_t[batch - 1, M_pad - 1, N_pad - 1]) / QKV_t[b, i, N], name = "QKV")

    s = te.create_schedule(QKV.op)

    s[Qtensor_pad].parallel(s[Qtensor_pad].fuse(s[Qtensor_pad].op.axis[0], s[Qtensor_pad].op.axis[1]))
    s[Ktensor_pad].parallel(s[Ktensor_pad].fuse(s[Ktensor_pad].op.axis[0], s[Ktensor_pad].op.axis[1]))
    s[Vtensor_pad].parallel(s[Vtensor_pad].fuse(s[Vtensor_pad].op.axis[0], s[Vtensor_pad].op.axis[1]))
    s[QKV].parallel(s[QKV].fuse(s[QKV].op.axis[0], s[QKV].op.axis[1]))

    tile_M = [-1, TILE1_M, MICRO_M]
    tile_N = [-1, TILE1_N, MICRO_N]

    # QKV_t
    b, i, j = s[QKV_t].op.axis
    i1, i2, i3 = tile_axes(s, QKV_t, i, tile_M)
    j1, j2, j3 = tile_axes(s, QKV_t, j, tile_N)
    tile_K = [-1, TILE1_K, L_pad]
    l1, l2, l3 = tile_axes(s, QKV_t, l, [-1, 1, TILE1_N*MICRO_N])
    s[QKV_t].reorder(b, i1, l1, j1, j2, i2, l2, i3, j3, l3)
    s[QKV_t].tensorize(i3, intrin_micro_kernel_gemm2(dtype, TILE1_N*MICRO_N))
    bi1 = s[QKV_t].fuse(b, i1)
    s[QKV_t].parallel(bi1)
    s[QKV_t].pragma(bi1, "import_llvm", kernels_impl(K_pad, TILE1_N*MICRO_N))

    # QK_exp
    s[QK_exp].compute_at(s[QKV_t], l1)

    b, i, l = s[QK_exp].op.axis
    i1, i2, i3 = tile_axes(s, QK_exp, i, tile_M)
    l1, l2, l3 = tile_axes(s, QK_exp, l, tile_N)
    s[QK_exp].reorder(b, i1, l1, l2, i2, i3, l3)
    s[QK_exp].tensorize(i3, intrin_micro_kernel_exp(dtype))
    # bi1 = s[QK_exp].fuse(b, i1)
    # s[QK_exp].parallel(bi1)

    # QK_pad
    s[QK_pad].compute_at(s[QK_exp], l1)

    b, i, l = s[QK_pad].op.axis
    i1, i2, i3 = tile_axes(s, QK_pad, i, tile_M)
    l1, l2, l3 = tile_axes(s, QK_pad, l, tile_N)
    tile_K = [-1, TILE1_K, K_pad]
    k1, k2, k3 = tile_axes(s, QK_pad, k, tile_K)
    s[QK_pad].reorder(b, i1, l1, k1, k2, i2, l2, i3, l3, k3)
    s[QK_pad].tensorize(i3, intrin_micro_kernel_gemm1(dtype, K_pad))
    # bi1 = s[QK_pad].fuse(b, i1)
    # s[QK_pad].parallel(bi1)

    return s, [Qtensor, Ktensor, Vtensor, QKV]

def test_shape(batch, M, K, L, N):
    s, arg_bufs = gemm_chain(batch, M, K, L, N, "float32")

    [Qtensor, Ktensor, Vtensor,  QKV] = arg_bufs

    print(tvm.lower(s, [Qtensor, Ktensor, Vtensor, QKV], simple_mode = True))
    func = tvm.build(s, arg_bufs, name = "gemm_chain")
    
    def softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    Q_np = np.random.uniform(size=(batch, M, K)).astype(np.float32)
    K_np = np.random.uniform(size=(batch, K, L)).astype(np.float32)
    V_np = np.random.uniform(size=(batch, L, N)).astype(np.float32)
    QKV_np = np.random.uniform(size=(batch, M, N)).astype(np.float32)
    for i in range(batch):
        QKV_np[i] = softmax(Q_np[i].dot(K_np[i]), 1).dot(V_np[i])

    dev = tvm.cpu()
    Q_tvm = tvm.nd.array(Q_np, device=dev)
    K_tvm = tvm.nd.array(K_np, device=dev)
    V_tvm = tvm.nd.array(V_np, device=dev)
    QKV_tvm = tvm.nd.empty((batch, M, N), device=dev)

    func(Q_tvm, K_tvm, V_tvm, QKV_tvm)

    # Check results
    np.testing.assert_allclose(QKV_np, QKV_tvm.numpy(), rtol=2e-3)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, dev, number=10000)
    cost = np.median(evaluator(Q_tvm, K_tvm, V_tvm, QKV_tvm).results)
    print("Execution time of this operator: %.3f ms" % (cost * 1000))
    return cost

if __name__=="__main__":
    re_file = open("gemm_softmax_gemm_x86_result.log", "w")
    # large
    print(test_shape(16, 512, 64, 512, 64), file = re_file)
    # base
    print(test_shape(12, 512, 64, 512, 64), file = re_file)
    # medium
    print(test_shape(8, 512, 64, 512, 64), file = re_file)
    # small
    print(test_shape(8, 512, 64, 512, 64), file = re_file)
    # mini
    print(test_shape(4, 512, 64, 512, 64), file = re_file)
    # tiny
    print(test_shape(2, 512, 64, 512, 64), file = re_file)
