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
        ret = f"PackedIntrinsic(;"
        ret += f"    load={self.load_intrinsics};"
        ret += f"    compute={self.compute_intrinsic};"
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

def gemm_impl(shape, isa, dtype, prefix):
    assert len(shape) == 3
    MI, NI, KI = shape
    '''
    float32
                MI     NI      KI
    avx512      6       16      *
    avx256      6       64/32   *
    '''
    assert MI == 6
    if isa == "avx2" and dtype == "float32":
        assert NI == 16
    if isa == "avx512" and dtype == "float32":
        assert NI == 32 or NI == 64
    if isa == "avx2" and dtype == "float64":
        assert NI == 8
    if isa == "avx512" and dtype == "float64":
        assert NI == 16 or NI == 64
    cc_code = """
        #include <immintrin.h>
        #include <stdio.h>

        extern "C" int %s_update(__TYPE__ *C, __TYPE__ *A, __TYPE__ *B, int K_, int N_, int c_stride_) {
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
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__4;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__5;"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__6;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__7;"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__8;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__9;"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__10;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__11;"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__12;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__13;"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ 0(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__14;"
                "__MOVInst__ __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__), %%%%__regName__15;"

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
                "__MOVInst__ 0(%%%%rbx), %%%%__regName__2;"
                "__MOVInst__ __nBytePerReg__(%%%%rbx), %%%%__regName__3;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r8, __TypeLen__), %%%%__regName__0;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r9, __TypeLen__), %%%%__regName__1;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__0, %%%%__regName__4;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__0, %%%%__regName__5;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__1, %%%%__regName__6;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__1, %%%%__regName__7;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r10, __TypeLen__), %%%%__regName__0;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r11, __TypeLen__), %%%%__regName__1;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__0, %%%%__regName__8;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__0, %%%%__regName__9;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__1, %%%%__regName__10;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__1, %%%%__regName__11;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r12, __TypeLen__), %%%%__regName__0;"
                "__BDCASTInst__ 0(%%%%rax, %%%%r13, __TypeLen__), %%%%__regName__1;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__0, %%%%__regName__12;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__0, %%%%__regName__13;"
                "__FMAInst__ %%%%__regName__2, %%%%__regName__1, %%%%__regName__14;"
                "__FMAInst__ %%%%__regName__3, %%%%__regName__1, %%%%__regName__15;"

                "lea __TypeLen__(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, __TypeLen__), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "__MOVInst__ %%%%__regName__4, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__5, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ %%%%__regName__6, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__7, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ %%%%__regName__8, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__9, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ %%%%__regName__10, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__11, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ %%%%__regName__12, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__13, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
                "add %%%%rdx, %%%%r8;"
                "__MOVInst__ %%%%__regName__14, 0(%%%%rcx, %%%%r8, __TypeLen__);"
                "__MOVInst__ %%%%__regName__15, __nBytePerReg__(%%%%rcx, %%%%r8, __TypeLen__);"
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
        extern "C" int %s_reset(__TYPE__ *cc, int stride) {
            #pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    cc[i*stride+j] = 0.0;
                }
            return 0;
        }
        extern "C" int %s_update_baseline(float * C, float *A, float*B, int K_, int N_, int c_stride_){
            for (int k = 0; k < %d; k++){
                for (int i = 0; i < %d; i++){
                    for (int j = 0; j < %d; j++){
                        C[i * c_stride_ + j] += A[i * K_ + k] * B[k * N_ + j];
                    }
                }
            }
            return 0;
        }
    """ % (
        prefix,
        KI,
        prefix,
        MI,
        NI,
        prefix,
        KI,MI,NI
    )

    if isa == "avx2":
        cc_code = cc_code.replace("__regName__", "ymm")
        cc_code = cc_code.replace("__nBytePerReg__", "32")
    elif isa == "avx512":
        cc_code = cc_code.replace("__regName__", "zmm")
        cc_code = cc_code.replace("__nBytePerReg__", "64")
    if dtype == "float32":
        cc_code = cc_code.replace("__TypeLen__", "4")
        cc_code = cc_code.replace("__MOVInst__", "vmovups")
        cc_code = cc_code.replace("__FMAInst__", "vfmadd231ps")
        cc_code = cc_code.replace("__BDCASTInst__", "vbroadcastss")
        cc_code = cc_code.replace("__TYPE__", "float")
    elif dtype == "float64":
        cc_code = cc_code.replace("__TypeLen__", "8")
        cc_code = cc_code.replace("__MOVInst__", "vmovupd")
        cc_code = cc_code.replace("__FMAInst__", "vfmadd231pd")
        cc_code = cc_code.replace("__BDCASTInst__", "vbroadcastsd")
        cc_code = cc_code.replace("__TYPE__", "double")
    if isa == "avx512" and dtype == "float32" and NI == 64:
        cc_code = '''
#include <unistd.h>
#include <stdio.h>
extern "C" int %s_update(float *C, float *A, float *B, int K_, int N_, int c_stride_)
{
    getpid();
    long long K = K_;
    long long N = N_;
    long long c_stride = c_stride_;

    __asm__ __volatile__(
        // AT&T syntax: src dst
        "mov %%[A], %%%%rax;"
        "mov %%[B], %%%%rbx;"
        "mov %%[C], %%%%rcx;"
        "mov %%[K], %%%%rsi;"
        "mov %%[N], %%%%rdi;"

        "mov %%[c_stride], %%%%rdx;"
        "xor %%%%r8, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm0;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm1;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm2;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm3;"
        "add %%%%rdx, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm4;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm5;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm6;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm7;"
        "add %%%%rdx, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm8;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm9;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm10;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm11;"
        "add %%%%rdx, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm12;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm13;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm14;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm15;"
        "add %%%%rdx, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm16;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm17;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm18;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm19;"
        "add %%%%rdx, %%%%r8;"
        "vmovups 0(%%%%rcx, %%%%r8, 4), %%%%zmm20;"
        "vmovups 64(%%%%rcx, %%%%r8, 4), %%%%zmm21;"
        "vmovups 128(%%%%rcx, %%%%r8, 4), %%%%zmm22;"
        "vmovups 192(%%%%rcx, %%%%r8, 4), %%%%zmm23;"

        "mov %%[K], %%%%rdx;"
        "mov $0, %%%%r8;"
        "lea (%%%%r8, %%%%rdx), %%%%r9;"
        "lea (%%%%r9, %%%%rdx), %%%%r10;"
        "lea (%%%%r10, %%%%rdx), %%%%r11;"
        "lea (%%%%rax, %%%%r11, 4), %%%%r11;"

        "mov $%d, %%%%rdx;"
        "test %%%%rdx, %%%%rdx;"
        "jz .store%%=;"

        ".compute%%=:"
        "vmovups 0(%%%%rbx), %%%%zmm28;"
        "vmovups 64(%%%%rbx), %%%%zmm29;"
        "vmovups 128(%%%%rbx), %%%%zmm30;"
        "vmovups 192(%%%%rbx), %%%%zmm31;"
        "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%zmm25;"
        "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%zmm26;"
        "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%zmm27;"
        "vfmadd231ps %%%%zmm25, %%%%zmm28, %%%%zmm0;"
        "vfmadd231ps %%%%zmm25, %%%%zmm29, %%%%zmm1;"
        "vfmadd231ps %%%%zmm25, %%%%zmm30, %%%%zmm2;"
        "vfmadd231ps %%%%zmm25, %%%%zmm31, %%%%zmm3;"
        "vfmadd231ps %%%%zmm26, %%%%zmm28, %%%%zmm4;"
        "vfmadd231ps %%%%zmm26, %%%%zmm29, %%%%zmm5;"
        "vfmadd231ps %%%%zmm26, %%%%zmm30, %%%%zmm6;"
        "vfmadd231ps %%%%zmm26, %%%%zmm31, %%%%zmm7;"
        "vfmadd231ps %%%%zmm27, %%%%zmm28, %%%%zmm8;"
        "vfmadd231ps %%%%zmm27, %%%%zmm29, %%%%zmm9;"
        "vfmadd231ps %%%%zmm27, %%%%zmm30, %%%%zmm10;"
        "vfmadd231ps %%%%zmm27, %%%%zmm31, %%%%zmm11;"

        "vbroadcastss 0(%%%%r11, %%%%r8, 4), %%%%zmm25;"
        "vbroadcastss 0(%%%%r11, %%%%r9, 4), %%%%zmm26;"
        "vbroadcastss 0(%%%%r11, %%%%r10, 4), %%%%zmm27;"
        "vfmadd231ps %%%%zmm25, %%%%zmm28, %%%%zmm12;"
        "vfmadd231ps %%%%zmm25, %%%%zmm29, %%%%zmm13;"
        "vfmadd231ps %%%%zmm25, %%%%zmm30, %%%%zmm14;"
        "vfmadd231ps %%%%zmm25, %%%%zmm31, %%%%zmm15;"
        "vfmadd231ps %%%%zmm26, %%%%zmm28, %%%%zmm16;"
        "vfmadd231ps %%%%zmm26, %%%%zmm29, %%%%zmm17;"
        "vfmadd231ps %%%%zmm26, %%%%zmm30, %%%%zmm18;"
        "vfmadd231ps %%%%zmm26, %%%%zmm31, %%%%zmm19;"
        "vfmadd231ps %%%%zmm27, %%%%zmm28, %%%%zmm20;"
        "vfmadd231ps %%%%zmm27, %%%%zmm29, %%%%zmm21;"
        "vfmadd231ps %%%%zmm27, %%%%zmm30, %%%%zmm22;"
        "vfmadd231ps %%%%zmm27, %%%%zmm31, %%%%zmm23;"

        "lea 4(%%%%rax), %%%%rax;"
        "lea 4(%%%%r11), %%%%r11;"
        "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
        "sub $1, %%%%rdx;"
        "jnz .compute%%=;"

        ".store%%=:"
        "mov %%[c_stride], %%%%rdx;"
        "xor %%%%r8, %%%%r8;"
        "vmovups %%%%zmm0, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm1, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm2, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm3, 192(%%%%rcx, %%%%r8, 4);"
        "add %%%%rdx, %%%%r8;"
        "vmovups %%%%zmm4, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm5, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm6, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm7, 192(%%%%rcx, %%%%r8, 4);"
        "add %%%%rdx, %%%%r8;"
        "vmovups %%%%zmm8, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm9, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm10, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm11, 192(%%%%rcx, %%%%r8, 4);"
        "add %%%%rdx, %%%%r8;"
        "vmovups %%%%zmm12, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm13, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm14, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm15, 192(%%%%rcx, %%%%r8, 4);"
        "add %%%%rdx, %%%%r8;"
        "vmovups %%%%zmm16, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm17, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm18, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm19, 192(%%%%rcx, %%%%r8, 4);"
        "add %%%%rdx, %%%%r8;"
        "vmovups %%%%zmm20, 0(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm21, 64(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm22, 128(%%%%rcx, %%%%r8, 4);"
        "vmovups %%%%zmm23, 192(%%%%rcx, %%%%r8, 4);"
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
extern "C" int %s_reset(float *cc, int stride) {
    #pragma unroll
    for (int i = 0; i < %d; ++i) 
        #pragma unroll
        for (int j = 0; j < %d; ++j) {
            cc[i*stride+j] = 0.0;
        }
    return 0;
}
extern "C" int %s_update_baseline(float * C, float *A, float*B, int K_, int N_, int c_stride_){
    for (int k = 0; k < %d; k++){
        for (int i = 0; i < %d; i++){
            for (int j = 0; j < %d; j++){
                C[i * c_stride_ + j] += A[i * K_ + k] * B[k * N_ + j];
            }
        }
    }
    return 0;
}
        '''%(prefix, KI, prefix, MI, NI, prefix, KI, MI, NI)
    from tvm.contrib import utils, clang
    with open("gemm.cc", 'w') as f:
        f.write(cc_code) 
    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(
        cc_code, output=ll_path, options=["-mavx2", "-mavx512f", "-msse"]
    )
    return ll_code

def intrin_gemm(
    shape, dtype, prefix, baseline = False
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
                    prefix + "_update" + ("_baseline" if baseline else ""),
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

def intrin_gemm_noreshape(
    shape, dtype, prefix, baseline = False
):
    assert (len(shape) == 3)
    MICRO_M, MICRO_N, MICRO_K = shape 
    A = te.placeholder((1, 1, MICRO_M, 1, MICRO_K), name="A", dtype=dtype)
    B = te.placeholder((1, 1, MICRO_K, 1, MICRO_N), name="B", dtype=dtype)
    ko = te.reduce_axis((0, 1), "ko")
    ki = te.reduce_axis((0, MICRO_K), name="ki")
    C = te.compute(
        (1, 1, MICRO_M, 1, MICRO_N), lambda b, mo, mi, no, ni: te.sum(A[b, mo, mi, ko, ki] * B[b, ko, ki, no, ni], axis=[ko, ki]), name="C"
    )
    aStride = te.var("astride")
    bStride = te.var("bstride")
    cStride = te.var("cstride")
    Ab = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name=prefix + "A",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var(), te.var(), aStride, te.var(), te.var()],
    )
    Bb = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name=prefix + "B",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var(), te.var(), bStride, te.var(), te.var()],
    )
    Cb = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name=prefix + "C",
        data_alignment=64,
        offset_factor=1,
        strides=[te.var(), te.var(), cStride, te.var(), te.var()],
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    prefix + "_update" + ("_baseline" if baseline else ""),
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    aStride,
                    bStride,
                    cStride
                )
            )
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "int32", prefix + "_reset", cc.access_ptr("w"), cStride
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})

def conv_impl(shape, isa, dtype, prefix):
    assert len(shape) == 7
    N, K, H, W, C, R, S = shape
    assert N == 1
    assert H % 3 == 0
    assert K % 4 == 0
    assert R == 3
    assert C == 32 or C == 64 or C == 128
    assert S == 3
    if isa == 'avx512' and dtype == 'float32':
        assert W % 16 == 0
    if isa == 'avx2' and dtype == 'float32':
        assert W % 8 == 0
    if isa == 'avx512' and dtype == 'float64':
        assert W % 8 == 0
    if isa == 'avx2' and dtype == 'float64':
        assert W % 4 == 0
    cc_code = """
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#define KI __KI__
#define HI __HI__
#define CI __CI__
#define WI __WI__
extern "C" int __prefix__conv_update(__TYPE__ *In, __TYPE__ *Weight, __TYPE__ *Out, int HW_s_, int W_s_,
                                int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_)
{
    unsigned long long HW_s = HW_s_;
    unsigned long long W_s = W_s_;
    unsigned long long HRWS_s = HRWS_s_;
    unsigned long long WS_s = WS_s_;
    unsigned long long CRS_s = CRS_s_;
    unsigned long long RS_s = RS_s_;
    unsigned long long S_s = S_s_;
    getpid();
    
    for (int k = 0; k < __K__; k+= KI){
        for(int h = 0; h < __H__; h += HI){
             for (int c = 0; c < __C__; c += CI){
                for (int w = 0; w < __W__; w += WI){
                    __TYPE__ * Out_ = Out+k * HW_s_ + h * W_s_ + w;
                    __TYPE__ * In_ = In + c * HRWS_s_ + h * WS_s_ + w;
                    __TYPE__ * Weight_ = Weight + k *CRS_s_ + c * RS_s_;
                    __asm__ __volatile__(
                        "mov %[W], %%rax;"
                        "mov %[HW], %%rbx;"

                        "mov %[Out_], %%rdi;" // Out[k * HW]
                        "__MOVInst__ (%%rdi), %%__regName__4;"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__5;"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__6;"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__ (%%rdi), %%__regName__7;"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__8;"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__9;"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__ (%%rdi), %%__regName__10;"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__11;"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__12;"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__ (%%rdi), %%__regName__13;"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__14;"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ (%%rdi, %%rdx, __TypeLen__), %%__regName__15;"

                        // calculation
                        /*
                        for n in [0, 1): "eliminate"
                            for c in [0, C):
                                for r in [0, 3): # unroll

                                    for s in [0, 3): # unroll
                                        for h in [0, 3):#unroll
                                            __regName___h = In[c*(H+R-1)*(W+S-1)+(h+r)*(W+S-1)+(s)](0:7)
                                        for k in [0, __TypeLen__):#unroll
                                            __regName___3 = bdcast(Weight[k*CRS+c*RS+r*S+s])
                                            for h in [0, 3): #unroll
                                                fma(__regName___h, __regName___3, __regName___{3k+h+__TypeLen__})

                        regs:
                            values:
                                In[c*(H+R-1)*(W+S-1)]                           Base_In_c       %%rax   1
                                In[c0*(H+R-1)*(W+S-1)+r*(W+S-1)+s]              Base_In         %%rbx   0
                                In[c0*(H+R-1)*(W+S-1)+r0*(W+S-1)+s0+h*(W+S-1)]  hWS             %%rcx   0
                                Weight[c*RS]                                    Base_Weight_C   %%rdx   1
                                Weight[c0*RS+r*S+s]                             Base_Weight     %%rdi   0
                                Weight[c0*RS+r0*S+s0+k*CRS]                     kCRS            %%rsi   0
                                c                                                               %%r13   1
                            strides:
                                HRWS                                                            %%r8    1
                                WS                                                              %%r9    1
                                CRS                                                             %%r10   1
                                RS                                                              %%r11   1
                                S                                                               %%r12   1
                        */
                        "mov $__CI__, %%r13;"
                        "mov %[In_], %%rax;"
                        "mov %[Weight_], %%rdx;"
                        "mov %[HRWS], %%r8;"
                        "mov %[WS], %%r9;"
                        "mov %[CRS], %%r10;"
                        "mov %[RS], %%r11;"
                        "mov %[S], %%r12;"
                        ".compute%=:;"

                        "mov %%rax, %%rbx;"
                        "mov %%rdx, %%rdi;"

                        /*cr=0s=0(h)(kh) begin*/
                        "mov %%rbx, %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "mov %%rdi, %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=0(h)(kh) end*/

                        /*cr=0s=1(h)(kh) begin*/
                        "lea __TypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __TypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=1(h)(kh) end*/

                        /*cr=0s=2(h)(kh) begin*/
                        "lea __DoubleTypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __DoubleTypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=2(h)(kh) end*/

                        /*cr=1s=0(h)(kh) begin*/
                        "lea (%%rbx, %%r9, __TypeLen__), %%rbx;"
                        "mov %%rbx, %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea (%%rdi, %%r12, __TypeLen__), %%rdi;"
                        "mov %%rdi, %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=1s=0(h)(kh) end*/

                        /*cr=1s=1(h)(kh) begin*/
                        "lea __TypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __TypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=1s=1(h)(kh) end*/

                        /*cr=1s=2(h)(kh) begin*/
                        "lea __DoubleTypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __DoubleTypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=1s=2(h)(kh) end*/

                        /*cr=2s=0(h)(kh) begin*/
                        "lea (%%rbx, %%r9, __TypeLen__), %%rbx;"
                        "mov %%rbx, %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea (%%rdi, %%r12, __TypeLen__), %%rdi;"
                        "mov %%rdi, %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=0(h)(kh) end*/

                        /*cr=2s=1(h)(kh) begin*/
                        "lea __TypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __TypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=1(h)(kh) end*/

                        /*cr=2s=2(h)(kh) begin*/
                        "lea __DoubleTypeLen__(%%rbx), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__0;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__1;"
                        "lea (%%rcx, %%r9, __TypeLen__), %%rcx;"
                        "__MOVInst__ (%%rcx), %%__regName__2;"

                        "lea __DoubleTypeLen__(%%rdi), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__4;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__5;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__6;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__7;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__8;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__9;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__10;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__11;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__12;"

                        "lea (%%rsi, %%r10, __TypeLen__), %%rsi;"
                        "__BDCASTInst__ (%%rsi), %%__regName__3;"
                        "__FMAInst__ %%__regName__0, %%__regName__3, %%__regName__13;"
                        "__FMAInst__ %%__regName__1, %%__regName__3, %%__regName__14;"
                        "__FMAInst__ %%__regName__2, %%__regName__3, %%__regName__15;"
                        /*cr=0s=2(h)(kh) end*/

                        "lea (%%rax, %%r8, __TypeLen__), %%rax;"
                        "lea (%%rdx, %%r11, __TypeLen__), %%rdx;"

                        "sub $1, %%r13;"
                        "jnz .compute%=;"

                        // store Out
                        "mov %[W], %%rax;"
                        "mov %[HW], %%rbx;"

                        "mov %[Out_], %%rdi;" // Out[k * HW]
                        "__MOVInst__ %%__regName__4, (%%rdi);"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__5, (%%rdi, %%rdx, __TypeLen__);"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__6, (%%rdi, %%rdx, __TypeLen__);"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__  %%__regName__7, (%%rdi);"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__8, (%%rdi, %%rdx, __TypeLen__);"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__9, (%%rdi, %%rdx, __TypeLen__);"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__ %%__regName__10, (%%rdi);"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__11, (%%rdi, %%rdx, __TypeLen__);"
                        "add %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__12, (%%rdi, %%rdx, __TypeLen__);"

                        "lea (%%rdi, %% rbx, __TypeLen__), %%rdi;"
                        "__MOVInst__ %%__regName__13, (%%rdi);"
                        "mov %%rax, %%rdx;"
                        "__MOVInst__ %%__regName__14, (%%rdi, %%rdx, __TypeLen__);"
                        "add %%rax, %%rdx;"
                        "__MOVInst__  %%__regName__15, (%%rdi, %%rdx, __TypeLen__);"

                        :
                        : [In_] "m"(In_),
                        [Weight_] "m"(Weight_),
                        [Out_] "m"(Out_),
                        [HW] "m"(HW_s),
                        [W] "m"(W_s),
                        [HRWS] "m"(HRWS_s),
                        [WS] "m"(WS_s),
                        [CRS] "m"(CRS_s),
                        [RS] "m"(RS_s),
                        [S] "m"(S_s)
                        : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "__regName__0", "__regName__1", "__regName__2", "__regName__3", "__regName__4", "__regName__5", "__regName__6", "__regName__7", "__regName__8", "__regName__9", "__regName__10", "__regName__11", "__regName__12", "__regName__13", "__regName__14", "__regName__15");
                }
            }
        }
    }
    return 0;
}

extern "C" int __prefix__conv_reset(__TYPE__ *Out, int KHW, int HW, int W,
                          int N_, int K_, int H_, int W_)
{
    for (int n = 0; n < N_; n++)
        for (int k = 0; k < K_; k++)
            for (int h = 0; h < H_; h++)
                for (int w = 0; w < W_; w++)
                {
                    Out[n * KHW + k * HW + h * W + w] = 0;
                }
    return 0;
}
"""
    cc_code = cc_code.replace("__K__", f"{K}")
    cc_code = cc_code.replace("__H__", f"{H}")
    cc_code = cc_code.replace("__W__", f"{W}")
    cc_code = cc_code.replace("__C__", f"{C}")
    cc_code = cc_code.replace("__prefix__", prefix)

    CI = C if (C == 32 or C == 64) else 64
    assert C % CI == 0
    WI = 8 if isa == "avx2" else 16 if isa == "avx512" else None
    assert WI and W % WI == 0 
    cc_code = cc_code.replace("__KI__", f"{4}")
    cc_code = cc_code.replace("__HI__", f"{3}")
    cc_code = cc_code.replace("__CI__", f"{CI}")
    cc_code = cc_code.replace("__WI__", f"{WI}")
    
    if isa == "avx2":
        cc_code = cc_code.replace("__regName__", "ymm")
    elif isa == "avx512":
        cc_code = cc_code.replace("__regName__", "zmm")
    if dtype == "float32":
        cc_code = cc_code.replace("__TypeLen__", "4")
        cc_code = cc_code.replace("__MOVInst__", "vmovups")
        cc_code = cc_code.replace("__FMAInst__", "vfmadd231ps")
        cc_code = cc_code.replace("__BDCASTInst__", "vbroadcastss")
        cc_code = cc_code.replace("__TYPE__", "float")
        cc_code = cc_code.replace("__DoubleTypeLen__", "8")
    elif dtype == "float64":
        cc_code = cc_code.replace("__TypeLen__", "8")
        cc_code = cc_code.replace("__MOVInst__", "vmovupd")
        cc_code = cc_code.replace("__FMAInst__", "vfmadd231pd")
        cc_code = cc_code.replace("__BDCASTInst__", "vbroadcastsd")
        cc_code = cc_code.replace("__TYPE__", "double")
        cc_code = cc_code.replace("__DoubleTypeLen__", "16")
    
    from tvm.contrib import utils, clang
    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-mavx", "-msse"])
    return ll_code

def intrin_conv(shape, dtype, prefix):
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
    In_ = te.placeholder([N_, C_, H_ + R_ - 1, W_ + S_ - 1], name="in", dtype=dtype)
    Weight_ = te.placeholder([K_, C_, R_, S_], name="weight", dtype=dtype)
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
        data_alignment=64,
        offset_factor=1,
        strides=[vars["chrws"], vars["hrws"], vars["ws"], 1],
    )  # N C H+R-1 W+S-1
    Weight_b = tvm.tir.decl_buffer(
        Weight_.shape,
        Weight_.dtype,
        name="Weight",
        data_alignment=64,
        offset_factor=1,
        strides=[vars["crs"], vars["rs"], vars["s"], 1],
    )  # K C R S
    Out_b = tvm.tir.decl_buffer(
        Out_.shape,
        Out_.dtype,
        name="Out",
        data_alignment=64,
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
                    prefix + "conv_update",
                    In__.access_ptr("r"),
                    Weight__.access_ptr("r"),
                    Out__.access_ptr("w"),
                    vars["hw"],
                    vars["w"],
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


def intrin_exp(shape, dtype, prefix):
    assert len(shape) == 2
    MICRO_M, MICRO_N = shape
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
                prefix + "compute_exp",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                cc.strides[0], aa.strides[0]
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})

def intrin_exp_noreshape(shape, dtype, prefix):
    assert len(shape) == 2
    MICRO_M, MICRO_N = shape
    A = te.placeholder((1, 1, MICRO_M, 1, MICRO_N), name="A", dtype=dtype)
    C = te.compute((1, 1, MICRO_M, 1, MICRO_N), lambda b, mo, mi, no, ni: te.exp(A[b, mo, mi, no, ni]), name="C")
    aStride, cStride = te.var("aStride"), te.var("cStride")
    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="A", data_alignment=64, offset_factor=1, strides=[te.var(), te.var(), aStride, te.var(), te.var()])
    Cb = tvm.tir.decl_buffer(C.shape, C.dtype, name="C", data_alignment=64, offset_factor=1, strides=[te.var(), te.var(), cStride, te.var(), te.var()])

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]
        ib = tvm.tir.ir_builder.create()
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                prefix + "compute_exp",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                cc.strides[2], aa.strides[2]
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, C: Cb})


def exp_impl(shape, isa, dtype, prefix):
    MICRO_M, MICRO_N = shape
    assert isa == "avx512"
    cc_code = '''
        #include <unistd.h>
        #include <immintrin.h>
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

        extern "C" int %scompute_exp(float* cc_, float *aa_, int stride1_, int stride2_) {\
            getpid();
            long long stride1 = stride1_;
            long long stride2 = stride2_;
            // #pragma unroll
            for (int i = 0; i < %d; ++i){
                for(int j = 0; j < %d; ++j){
                    float* cc = cc_ + i * stride1_ + j * 16;
                    float* aa = aa_ + i * stride2_ + j * 16;
                    __asm__(
                    ".exp%%=:"
                        "mov %%[aa], %%%%rax;"
                        "mov %%[cc], %%%%rcx;"
                        "mov %%[_ps_exp_hi_p], %%%%rbx;"
                        "mov %%[_ps_one_p], %%%%rdx;"
                        "vmovups (%%%%rax), %%%%zmm0;"

                        // _mm_min_ps
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vminps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_max_ps
                        // "mov %%[_ps_exp_lo_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vmaxps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_LOG2EF_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm0, %%%%zmm1, %%%%zmm2;"

                        // _mm_add_ps
                        // "mov %%[_ps_0p5_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vaddps %%%%zmm2, %%%%zmm1, %%%%zmm2;"

                        // _mm_cvttps_epi32
                        "vcvttps2dq %%%%zmm2, %%%%zmm1;"
                        
                        // _mm_cvtepi32_ps
                        "vcvtdq2ps %%%%zmm1, %%%%zmm1;"
                        
                        // sub if greater than
                        "vcmpps $14, %%%%zmm2, %%%%zmm1, %%%%k1;"
                        // "mov %%[_ps_one_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vsubps %%%%zmm3, %%%%zmm1, %%%%zmm2;"
                        // "vmovups %%%%zmm2, %%%%zmm1{%%%%k1};"
                        // "vmovups %%%%zmm1, %%%%zmm2;"
                        "vblendmps %%%%zmm2, %%%%zmm1, %%%%zmm2 %%{%%%%k1%%};"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_exp_C1_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        // _mm_sub_ps
                        "vsubps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // _mm_mul_ps
                        // "mov %%[_ps_cephes_exp_C2_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        // _mm_sub_ps
                        "vsubps %%%%zmm0, %%%%zmm1, %%%%zmm0;"

                        // mul & add
                        // "mov %%[_ps_cephes_exp_p0_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p1_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p2_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p3_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p4_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_cephes_exp_p5_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        "vmulps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        "vaddps %%%%zmm1, %%%%zmm0, %%%%zmm1;"
                        // "mov %%[_ps_one_p], %%%%rdx;"
                        "vmovups (%%%%rdx), %%%%zmm3;"
                        "vaddps %%%%zmm1, %%%%zmm3, %%%%zmm1;"

                        "vcvttps2dq %%%%zmm2, %%%%zmm2;"
                        // "mov %%[_ps_0x7f_p], %%%%rbx;"
                        "lea 64(%%%%rbx), %%%%rbx;"
                        "vmovups (%%%%rbx), %%%%zmm0;"
                        "vpaddd %%%%zmm2, %%%%zmm0, %%%%zmm2;"
                        "vpslld $23, %%%%zmm2, %%%%zmm2;"
                        "vmulps %%%%zmm2, %%%%zmm1, %%%%zmm1;"

                        "vmovups %%%%zmm1, (%%%%rcx);"
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
            getpid();
            return 0;
        }
    ''' % (prefix, MICRO_M, MICRO_N // 16)
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-mavx", "-mavx512f", "-msse"])
    return ll_code


def cpu_intrin(op, shape, isa, dtype, prefix="", baseline = False):
    loads = []
    if op == "gemm":
        compute = intrin_gemm(
            shape, dtype=dtype, prefix=prefix, baseline = baseline
        )
        code = gemm_impl(shape, isa, dtype, prefix)
    elif op == "gemm_noreshape":
        compute = intrin_gemm_noreshape(
            shape, dtype = dtype, prefix = prefix, baseline = baseline
        )
        code = gemm_impl(shape, isa, dtype, prefix)
    elif op == "conv":
        compute = intrin_conv(
            shape, dtype, prefix
        )
        code = conv_impl(shape, isa, dtype, prefix)
    elif op == "exp":
        compute = intrin_exp(
            shape, dtype, prefix
        )
        code = exp_impl(shape, isa, dtype, prefix)
    elif op == "exp_noreshape":
        compute = intrin_exp_noreshape(
            shape, dtype, prefix 
        )
        code = exp_impl(shape, isa, dtype, prefix)
    store = None
    pintrin = packed_intrinsic(
        loads, compute, store, ["global", "global"], "global", "global"
    )
    return pintrin, code