import tvm
import tvm._ffi
from . import _ffi_api
from tvm.runtime import Object
from typing import List, Dict
import tvm.te as te


@tvm._ffi.register_object("ditto.auto_tensorize.PackedIntrinsic")
class PackedIntrinsic(Object):
    """PackedIntrinsic object"""

    def __str__(self):
        ret = f"PackedIntrinsic(\n"
        ret += f"    load={self.load_intrinsics}\n"
        ret += f"    compute={self.compute_intrinsic}\n"
        ret += f"    store={self.store_intrinsic})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def packed_intrinsic(
    loads: List[tvm.te.tensor_intrin.TensorIntrin],
    compute: tvm.te.tensor_intrin.TensorIntrin,
    store: tvm.te.tensor_intrin.TensorIntrin,
    load_scopes: List[str],
    compute_scope: str,
    store_scope: str,
):
    return _ffi_api.PackedIntrinsic(
        loads, compute, store, load_scopes, compute_scope, store_scope
    )


def intrinsic_match(
    target: tvm.te.Tensor, packed_intrin: PackedIntrinsic, restrictions: List[str] = []
):
    innermost = False
    if "InnerMost" in restrictions:
        innermost = True
    samerange = False
    if "SameRange" in restrictions:
        samerange = True
    return _ffi_api.MatchIntrinsic(
        target, packed_intrin.compute_intrinsic.op, innermost, samerange
    )


# CUDA WMMA Intrinsic


def intrin_wmma_load_matrix_a(MI, NI, KI, in_dtype):
    A = tvm.te.placeholder((MI, KI), name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=int(KI * 2), offset_factor=256
    )
    C = tvm.te.compute((MI, KI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_a",
        data_alignment=int(KI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                MI,
                NI,
                KI,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                KI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="load_a"
    )


def intrin_wmma_load_matrix_b(MI, NI, KI, in_dtype):
    A = tvm.te.placeholder((KI, NI), name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=int(NI * 2), offset_factor=256
    )
    C = tvm.te.compute((KI, NI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_b",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                MI,
                NI,
                KI,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                NI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="load_b"
    )


def intrin_wmma_gemm(MI, NI, KI, in_dtype, acc_dtype):
    A = tvm.te.placeholder((MI, KI), name="A", dtype=in_dtype)
    B = tvm.te.placeholder((KI, NI), name="B", dtype=in_dtype)
    k = tvm.te.reduce_axis((0, KI), name="k")
    C = tvm.te.compute(
        (MI, NI),
        lambda ii, jj: tvm.te.sum(
            A[ii, k].astype(acc_dtype) * B[k, jj].astype(acc_dtype), axis=k
        ),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        data_alignment=int(KI * 2),
        offset_factor=256,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    MI,
                    NI,
                    KI,
                    BC.elem_offset // 256,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, B: BB, C: BC}, name="gemm"
    )


def intrin_wmma_store_matrix(MI, NI, KI, scope, acc_dtype):
    A = tvm.te.placeholder((MI, NI), name="A", dtype=acc_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        data_alignment=int(NI * 2),
        offset_factor=256,
    )
    C = tvm.te.compute((MI, NI), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, data_alignment=int(NI * 2), offset_factor=256
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                MI,
                NI,
                KI,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                NI,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, C: BC}, name="store"
    )


def cuda_wmma(
    M=16, N=16, K=16, in_dtype="float16", out_dtype="float32", scope="global"
):
    loads = [
        intrin_wmma_load_matrix_a(M, N, K, in_dtype),
        intrin_wmma_load_matrix_b(M, N, K, in_dtype),
    ]

    compute = intrin_wmma_gemm(M, N, K, in_dtype, out_dtype)

    store = intrin_wmma_store_matrix(M, N, K, scope, out_dtype)

    pintrin = packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )
    return pintrin

def gemm_impl(shape, prefix, instructionSet):
    assert len(shape) == 3
    MI, NI, KI = shape
    cc_code = """
        #include <immintrin.h>
        #include <stdio.h>

        extern "C" int %s_update(float *C, float *A, float *B, int K_, int N_, int c_stride_) {
            long long K = K_;
            long long N = N_;
            long long c_stride = c_stride_;
            __asm__(
                // AT&T syntax: src dst 
                "mov %%[A], %%%%rax;"
                "mov %%[B], %%%%rbx;"
                "mov %%[C], %%%%rcx;"
                "mov %%[K], %%%%rsi;"
                "mov %%[N], %%%%rdi;"

                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__4;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__5;"
                "add %%%%rdx, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__6;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__7;"
                "add %%%%rdx, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__8;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__9;"
                "add %%%%rdx, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__10;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__11;"
                "add %%%%rdx, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__12;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__13;"
                "add %%%%rdx, %%%%r8;"
                "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%__regName__14;"
                "vmovups __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__15;"

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
                "vmovups 0(%%%%rbx), %%%%__regName__2;"
                "vmovups __nBytePerReg__(%%%%rbx), %%%%__regName__3;"
                "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__4;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__5;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__6;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__7;"
                "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r11, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__8;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__9;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__10;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__11;"
                "vbroadcastss 0(%%%%rax, %%%%r12, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r13, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__12;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__13;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__14;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__15;"

                "lea 4(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovups %%%%__regName__4, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__5, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovups %%%%__regName__6, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__7, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovups %%%%__regName__8, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__9, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovups %%%%__regName__10, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__11, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovups %%%%__regName__12, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__13, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovups %%%%__regName__14, 0(%%%%rcx, %%%%r8, 4);"
                "vmovups %%%%__regName__15, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                :
                :[A] "m" (A),
                [B] "m" (B),
                [C] "m" (C),
                [K] "m" (K),
                [N] "m" (N),
                [c_stride] "m" (c_stride)
                :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "__regName__0", "__regName__1", "__regName__2", "__regName__3", "__regName__4", "__regName__5", "__regName__6", "__regName__7", "__regName__8", "__regName__9", "__regName__10", "__regName__11", "__regName__12", "__regName__13", "__regName__14", "__regName__15"
            );
            return 0;
        }
        extern "C" int %s_reset(float *cc, int stride) {
            #pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    cc[i*stride+j] = 0.0;
                }
            return 0;
        }

    """ % (
        prefix,
        KI,
        prefix,
        MI,
        NI
    )
    if instructionSet == "avx2":
        cc_code = cc_code.replace("__regName__", "ymm")
        cc_code = cc_code.replace("__nBytePerReg__", "32")
    elif instructionSet == "avx512":
        cc_code = cc_code.replace("__regName__", "zmm")
        cc_code = cc_code.replace("__nBytePerReg__", "64")
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(
        cc_code, output=ll_path, options=["-mavx", "-mavx512f", "-msse"]
    )
    return ll_code

def intrin_gemm(
    shape, dtype="float32", prefix=""
):
    assert (len(shape) == 3)
    MICRO_M, MICRO_N, MICRO_K = shape 
    a = te.placeholder((MICRO_M, MICRO_K), name="a", dtype=dtype)
    b = te.placeholder((MICRO_K, MICRO_N), name="b", dtype=dtype)
    k = te.reduce_axis((0, MICRO_K), name="k")
    c = te.compute(
        (MICRO_M, MICRO_N), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c"
    )
    Ab = tvm.tir.decl_buffer(
        a.shape,
        a.dtype,
        name=prefix + "A",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var("s1"), 1],
    )
    Bb = tvm.tir.decl_buffer(
        b.shape,
        b.dtype,
        name=prefix + "B",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var("s2"), 1],
    )
    Cb = tvm.tir.decl_buffer(
        c.shape,
        c.dtype,
        name=prefix + "C",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var("s3"), 1],
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    prefix + "_update",
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    aa.strides[0],
                    bb.strides[0],
                    cc.strides[0],
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32", prefix + "_reset", cc.access_ptr("w"), cc.strides[0]
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def conv_impl(prefix):
    cc_code = """
#pragma once 
extern "C" int conv_update(float *In, float *Weight, float *Out, int K, int H, int W, int C, int R, int S,
                           int N_, int K_, int H_, int W_, int C_, int R_, int S_)
{
    for (int n = 0; n < N_; n++)
        for (int h = 0; h < H_; h++)
            for (int c = 0; c < C_; c++)
                for (int r = 0; r < R_; r++)
                    for (int s = 0; s < S_; s++)
                        for (int k = 0; k < K_; k++)
                            for (int w = 0; w < W_; w++) // vectorize 8 / 16
                            {
                                Out[n * K * H * W + k * H * W + h * W + w] +=
                                    In[n * C * (H + R - 1) * (W + S - 1) + c * (H + R - 1) * (W + S - 1) + (h + r) * (W + S - 1) + (w + s)] * Weight[k * C * R * S + c * R * S + r * S + s];
                            }
    return 0;
}

/*
    for (long long n = 0; n < 1; n++)
        for (long long c = 0; c < C_in; c++) 
            for (long long r = 0; r < 3; r++) unroll
                for (long long s = 0; s < 3; s++) unroll
                    for (long long k = 0; k < 4; k++) unroll
                        for (long long h = 0; h < 3; h++) vectorize 
                            for (long long w = 0; w < 8; w++) vectorize 
                            {
                                Out[k * HW + h * W + w] +=
                                    In[c * HRWS_s + (h + r) * WS + (w + s)] * Weight[k *CRS + c * RS + r * S + s];
                            }   
*/
extern "C" int %sconv_update_avx2(float *In, float *Weight, float *Out, int HW_s_, int W_s_, int C_in_, 
int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_){
    long long HW_s = HW_s_;
    long long W_s = W_s_;
    long long C_in = C_in_;
    long long HRWS_s = HRWS_s_;
    long long WS_s = WS_s_;
    long long CRS_s = CRS_s_;
    long long RS_s = RS_s_;
    long long S_s = S_s_;
    __asm__(
        "mov %%[W], %%%%rax;"
        "mov %%[HW], %%%%rbx;"

        "mov %%[Out], %%%%rdi;" // Out[k * HW]
        "vmovups (%%%%rdi), %%%%ymm4;"
        "mov %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm5;"
        "add %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm6;"
        
        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups (%%%%rdi), %%%%ymm7;"
        "mov %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm8;"
        "add %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm9;"
        
        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups (%%%%rdi), %%%%ymm10;"
        "mov %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm11;"
        "add %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm12;"

        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups (%%%%rdi), %%%%ymm13;"
        "mov %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm14;"
        "add %%%%rax, %%%%rdx;"
        "vmovups (%%%%rdi, %%%%rdx, 4), %%%%ymm15;"

        // calculation
        /*
        for n in [0, 1): "eliminate"
            for c in [0, C):
                for r in [0, 3): # unroll
                    
                    for s in [0, 3): # unroll
                        for h in [0, 3):#unroll
                            ymm_h = In[c*(H+R-1)*(W+%%%%r12-1)+(h+r)*(W+%%%%r12-1)+(s)](0:7)
                        for k in [0, 4):#unroll
                            ymm_3 = bdcast(Weight[k*%%%%r10+c*%%%%r11+r*%%%%r12+s])
                            for h in [0, 3): #unroll
                                fma(ymm_h, ymm_3, ymm_{3k+h+4}) 

        regs:                                                                            
            values:
                In[c*(H+R-1)*(W+S-1)]                           Base_In_c       %%%%rax   1
                In[c0*(H+R-1)*(W+S-1)+r*(W+S-1)+s]              Base_In         %%%%rbx   0
                In[c0*(H+R-1)*(W+S-1)+r0*(W+S-1)+s0+h*(W+S-1)]  hWS             %%%%rcx   0
                Weight[c*RS]                                    Base_Weight_C   %%%%rdx   1
                Weight[c0*RS+r*S+s]                             Base_Weight     %%%%rdi   0
                Weight[c0*RS+r0*S+s0+k*CRS]                     kCRS            %%%%rsi   0
                c                                                               %%%%r13   1
            strides:
                HRWS                                                            %%%%r8    1
                WS                                                              %%%%r9    1
                CRS                                                             %%%%r10   1
                RS                                                              %%%%r11   1
                S                                                               %%%%r12   1
        */
        "mov %%[C_in], %%%%r13;"
        "mov %%[In], %%%%rax;"
        "mov %%[Weight], %%%%rdx;"
        "mov %%[HRWS], %%%%r8;"
        "mov %%[WS], %%%%r9;"
        "mov %%[CRS], %%%%r10;"
        "mov %%[RS], %%%%r11;"
        "mov %%[S], %%%%r12;"
    ".compute:;"

        "mov %%%%rax, %%%%rbx;"
        "mov %%%%rdx, %%%%rdi;"

        /*cr=0s=0(h)(kh) begin*/
        "mov %%%%rbx, %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "mov %%%%rdi, %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=0(h)(kh) end*/

        /*cr=0s=1(h)(kh) begin*/
        "lea 4(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 4(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=1(h)(kh) end*/

        /*cr=0s=2(h)(kh) begin*/
        "lea 8(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 8(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=2(h)(kh) end*/

        /*cr=1s=0(h)(kh) begin*/
        "lea (%%%%rbx, %%%%r9, 4), %%%%rbx;"
        "mov %%%%rbx, %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea (%%%%rdi, %%%%r12, 4), %%%%rdi;"
        "mov %%%%rdi, %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=1s=0(h)(kh) end*/

        /*cr=1s=1(h)(kh) begin*/
        "lea 4(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 4(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=1s=1(h)(kh) end*/

        /*cr=1s=2(h)(kh) begin*/
        "lea 8(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 8(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=1s=2(h)(kh) end*/

        /*cr=2s=0(h)(kh) begin*/
        "lea (%%%%rbx, %%%%r9, 4), %%%%rbx;"
        "mov %%%%rbx, %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea (%%%%rdi, %%%%r12, 4), %%%%rdi;"
        "mov %%%%rdi, %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=0(h)(kh) end*/

        /*cr=2s=1(h)(kh) begin*/
        "lea 4(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 4(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=1(h)(kh) end*/

        /*cr=2s=2(h)(kh) begin*/
        "lea 8(%%%%rbx), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm0;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm1;"
        "lea (%%%%rcx, %%%%r9, 4), %%%%rcx;"
        "vmovups (%%%%rcx), %%%%ymm2;"

        "lea 8(%%%%rdi), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm4;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm5;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm6;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm7;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm8;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm9;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm10;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm11;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm12;"

        "lea (%%%%rsi, %%%%r10, 4), %%%%rsi;"
        "vbroadcastss (%%%%rsi), %%%%ymm3;"
        "vfmadd231ps %%%%ymm0, %%%%ymm3, %%%%ymm13;"
        "vfmadd231ps %%%%ymm1, %%%%ymm3, %%%%ymm14;"
        "vfmadd231ps %%%%ymm2, %%%%ymm3, %%%%ymm15;"
        /*cr=0s=2(h)(kh) end*/

        "lea (%%%%rax, %%%%r8, 4), %%%%rax;"
        "lea (%%%%rdx, %%%%r11, 4), %%%%rdx;"

        "sub $1, %%%%r13;"
        "jnz .compute;"

// store Out
        "mov %%[W], %%%%rax;"
        "mov %%[HW], %%%%rbx;"

        "mov %%[Out], %%%%rdi;" // Out[k * HW]
        "vmovups %%%%ymm4, (%%%%rdi);"
        "mov %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm5, (%%%%rdi, %%%%rdx, 4);"
        "add %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm6, (%%%%rdi, %%%%rdx, 4);"
        
        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups  %%%%ymm7, (%%%%rdi);"
        "mov %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm8, (%%%%rdi, %%%%rdx, 4);"
        "add %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm9, (%%%%rdi, %%%%rdx, 4);"
        
        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups %%%%ymm10, (%%%%rdi);"
        "mov %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm11, (%%%%rdi, %%%%rdx, 4);"
        "add %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm12, (%%%%rdi, %%%%rdx, 4);"

        "lea (%%%%rdi, %%%% rbx, 4), %%%%rdi;"
        "vmovups %%%%ymm13, (%%%%rdi);"
        "mov %%%%rax, %%%%rdx;"
        "vmovups %%%%ymm14, (%%%%rdi, %%%%rdx, 4);"
        "add %%%%rax, %%%%rdx;"
        "vmovups  %%%%ymm15, (%%%%rdi, %%%%rdx, 4);"

        :
        :[In] "m" (In),
        [Weight] "m" (Weight),
        [Out] "m" (Out),
        [HW] "m" (HW_s),
        [W] "m" (W_s),
        [C_in] "m" (C_in),
        [HRWS] "m" (HRWS_s),
        [WS] "m" (WS_s),
        [CRS] "m" (CRS_s),
        [RS] "m" (RS_s),
        [S] "m" (S_s)
        :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
    );
    return 0;
}

extern "C" int %sconv_reset(float *Out, int K, int H, int W,
                           int N_, int K_, int H_, int W_)
{
    for (int n = 0; n < N_; n++)
        for (int k = 0; k < K_; k++)
            for (int h = 0; h < H_; h++)
                for (int w = 0; w < W_; w++)
                {
                    Out[n * K * H * W + k * H * W + h * W + w] = 0;
                }
    return 0;
}
    """ % (prefix, prefix)
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-mavx", "-msse"])
    return ll_code

def intrin_conv(shape, dtype = "float32", prefix = ""):
    """
        In = tvm.te.placeholder([N, C, H + R - 1, W + S - 1], dtype=in_dtype, name='In' + surfix)
    Weight = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name='Weight' + surfix)
    r = tvm.te.reduce_axis((0, R), name='rr' + surfix)
    s = tvm.te.reduce_axis((0, S), name='rs' + surfix)
    c = tvm.te.reduce_axis((0, C), name='rc' + surfix)
    Out = tvm.te.compute([N, K, H, W], lambda n,k,h,w:
        tvm.te.sum(In[n,c,h+r,w+s] * Weight[k,c,r,s], axis=[c,r,s]),
        name = 'Ou' + surfix
    )"""
    assert(len(shape) == 7)
    N_, K_, H_, W_, C_, R_, S_ = shape
    in_dtype = "float32"
    In_ = te.placeholder([N_, C_, H_ + R_ - 1, W_ + S_ - 1], name="in", dtype=in_dtype)
    Weight_ = te.placeholder([K_, C_, R_, S_], name="weight", dtype=in_dtype)
    r_ = tvm.te.reduce_axis((0, R_), name="rr")
    s_ = tvm.te.reduce_axis((0, S_), name="rs")
    co = tvm.te.reduce_axis((0, 1), name="rco")
    c_ = tvm.te.reduce_axis((0, C_), name="rci")
    Out_ = te.compute(
        [1, N_, 1, K_, 1, H_, 1, W_],
        lambda no, n_, ko, k_, ho, h_, wo, w_: te.sum(
            In_[n_, c_, h_ + r_, w_ + s_] * Weight_[k_, c_, r_, s_],
            axis=[co, c_, r_, s_],
        ),
        name="out",
    )
    #  [21119070, 158790, 402, 1],  value [350, 50, 10, 1],
    varnames = [
        "chrws",
        "hrws",
        "ws",
        "crs",
        "rs",
        "s",
        "khw",
        "hw",
        "w",
        "_nkhw",
        "_khw",
        "_hw",
        "_w",
    ]
    vars = {varname: tvm.te.var(varname) for varname in varnames}
    In_b = tvm.tir.decl_buffer(
        In_.shape,
        In_.dtype,
        name="In",
        offset_factor=1,
        strides=[vars["chrws"], vars["hrws"], vars["ws"], 1],
    )  # N C H+R-1 W+S-1
    Weight_b = tvm.tir.decl_buffer(
        Weight_.shape,
        Weight_.dtype,
        name="Weight",
        offset_factor=1,
        strides=[vars["crs"], vars["rs"], vars["s"], 1],
    )  # K C R S
    Out_b = tvm.tir.decl_buffer(
        Out_.shape,
        Out_.dtype,
        name="Out",
        offset_factor=1,
        strides=[
            vars["_nkhw"],
            vars["khw"],
            vars["_khw"],
            vars["hw"],
            vars["_hw"],
            vars["w"],
            vars["_w"],
            1,
        ],
    )  # N K H W

    def intrin_func(ins, outs):
        In__, Weight__ = ins
        Out__ = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            # extern "C" int conv_update_avx2(float *In, float *Weight, float *Out, int HW_s_, int W_s_, int C_in_,
            # int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_){
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    prefix + "conv_update_avx2",
                    In__.access_ptr("r"),
                    Weight__.access_ptr("r"),
                    Out__.access_ptr("w"),
                    vars["hw"],
                    vars["w"],
                    C_,
                    vars["hrws"],
                    vars["ws"],
                    vars["crs"],
                    vars["rs"],
                    vars["s"],
                )
                # extern "C" int conv_update(float *In, float *Weight, float *Out, int KHW, int HW, int W, int CHRWS, int HRWS, int WS, int CRS, int RS, int S,
                # int N_, int K_, int H_, int W_, int C_, int R_, int S_)
                # tvm.tir.call_extern(
                #     "int32",
                #     "conv_update",
                #     In__.access_ptr('r'),
                #     Weight__.access_ptr('r'),
                #     Out__.access_ptr('w'),
                #     vars['khw'], vars['hw'], vars['w'], vars['chrws'], vars['hrws'], vars['ws'], vars['crs'],vars['rs'],vars['s'],C_
                # )
            )
            return ib.get()

        def _reduce_reset():
            # long long conv_reset(float *Out, long long K, long long H, long long W,
            #            long long N_, long long K_, long long H_, long long W_)
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    prefix + "conv_reset",
                    Out__.access_ptr("w"),
                    vars["khw"],
                    vars["hw"],
                    vars["w"],
                    N_,
                    K_,
                    H_,
                    W_,
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(
        Out_.op, intrin_func, binds={In_: In_b, Weight_: Weight_b, Out_: Out_b}
    )

def cpu_intrin(op, shape, dtype="float32", prefix=""):
    loads = []
    if op == "gemm":
        compute = intrin_gemm(
            shape, dtype=dtype, prefix=prefix
        )
        code = gemm_impl(shape, prefix, "avx2")
    elif op == "conv":
        compute = intrin_conv(
            shape, dtype, prefix
        )
        code = conv_impl(prefix = prefix)
    store = None
    pintrin = packed_intrinsic(
        loads, compute, store, ["global", "global"], "global", "global"
    )
    return pintrin, code


