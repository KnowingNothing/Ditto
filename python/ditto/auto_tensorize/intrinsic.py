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


def intrin_micro_kernel_gemm2(
    dtype="float32", MICRO_M=16, MICRO_N=16, MICRO_K=16, prefix=""
):
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


def cpu_avx2(M=16, N=16, K=16, dtype="float32", prefix=""):
    loads = []
    compute = intrin_micro_kernel_gemm2(
        MICRO_M=M, MICRO_N=N, MICRO_K=K, dtype=dtype, prefix=prefix
    )
    store = None
    pintrin = packed_intrinsic(
        loads, compute, store, ["global", "global"], "global", "global"
    )
    return pintrin
