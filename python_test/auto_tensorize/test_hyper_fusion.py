import pytest
import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math

MI = 16
NI = 16
KI = 16
WARP_SIZE = 32
IN_VEC = 4
OUT_VEC = 4


def BatchGemmSoftmaxGemm(
    batch=12, M=512, N=64, K=64, L=512, in_dtype="float16", acc_dtype="float32"
):
    assert M % MI == 0
    assert N % NI == 0
    assert K % KI == 0
    assert L % NI == 0
    assert L % KI == 0

    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=in_dtype)

    A_shared = tvm.te.compute(
        [batch, M // MI, K // KI, MI, KI],
        lambda b, mo, ko, mi, ki: A[b, mo * MI + mi, ko * KI + ki],
        name="A_shared",
    )

    B_shared = tvm.te.compute(
        [batch, K // KI, L // NI, KI, NI],
        lambda b, ko, lo, ki, li: B[b, ko * KI + ki, lo * NI + li],
        name="B_shared",
    )

    rko = tvm.te.reduce_axis([0, K // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A_shared[b, mo, rko, mi, rki].astype(acc_dtype)
            * B_shared[b, rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    exp = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.exp(D_frag[b, mo, lo, mi, li]).astype(
            in_dtype
        ),
        name="exp",
    )

    ext_N = 2 ** math.ceil(math.log2(N // NI + 1)) * NI
    C_ext = tvm.te.compute(
        [batch, L // KI, ext_N // NI, KI, NI],
        lambda b, lo, no, li, ni: tvm.tir.if_then_else(
            no * NI + ni < N,
            C[b, lo * NI + li, no * NI + ni],
            tvm.tir.const(1, in_dtype),
        ),
        name="C_ext",
    )

    rlo = tvm.te.reduce_axis([0, L // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, ext_N // NI, MI, NI],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            exp[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_ext[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, n // NI, m % MI, n % NI].astype(in_dtype)
        / (
            E_frag[b, m // MI, ext_N // NI - 1, m % MI, NI - 1].astype(in_dtype)
            + tvm.tir.const(1e-5, in_dtype)
        ),
        name="F",
    )

    return [A, B, C], [F]


def intrin_wmma_load_matrix_a(in_dtype):
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


def intrin_wmma_load_matrix_b(in_dtype):
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


def intrin_wmma_gemm(in_dtype, acc_dtype):
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


def intrin_wmma_store_matrix(scope, acc_dtype):
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


def test_build_fusion_choice():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)
    print(fuse_choice)


def test_build_match_info():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("shared", "float32")

    first_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    first_match_info = at.match_info([mi, li, rki], first_packed)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("global", "float32")

    second_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    second_match_info = at.match_info([mi, ni, rli], second_packed)

    print(first_match_info)
    print(second_match_info)


def test_build_tensorize_hyper_fusion_state():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("shared", "float32")

    first_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    first_match_info = at.match_info([mi, li, rki], first_packed)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("global", "float32")

    second_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    second_match_info = at.match_info([mi, ni, rli], second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )
    print(tensorize_state)
    print(tensorize_state.summary(verbose=True))


def test_tensorize_cuda():
    ins, outs = BatchGemmSoftmaxGemm()
    A, B, C = ins
    (F,) = outs

    E_frag = F.op.input_tensors[0]
    exp, C_ext = E_frag.op.input_tensors
    D_frag = exp.op.input_tensors[0]
    A_shared, B_shared = D_frag.op.input_tensors

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    fuse_choice = at.fusion_choice(D_frag.op, E_frag.op, [b, m, n, rlo, mi, ni, rli], 3)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("shared", "float32")

    first_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, l, mi, li = D_frag.op.axis
    rko, rki = D_frag.op.reduce_axis
    first_match_info = at.match_info([mi, li, rki], first_packed)

    loads = [intrin_wmma_load_matrix_a("float16"), intrin_wmma_load_matrix_b("float16")]

    compute = intrin_wmma_gemm("float16", "float32")

    store = intrin_wmma_store_matrix("global", "float32")

    second_packed = at.packed_intrinsic(
        loads,
        compute,
        store,
        ["wmma.matrix_a", "wmma.matrix_b"],
        "wmma.accumulator",
        "wmma.accumulator",
    )

    b, m, n, mi, ni = E_frag.op.axis
    rlo, rli = E_frag.op.reduce_axis
    second_match_info = at.match_info([mi, ni, rli], second_packed)

    layer = ac.layer([F.op], inputs=[A, B, C])
    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {D_frag.op: first_match_info, E_frag.op: second_match_info}
    )

    V100 = hw.query_hw_param("gpu.cuda.V100")
    tensorize_param = at.cuda_tensorize_param(
        warp_size=32, ty_size=4, tz_size=2, input_vector_len=4, serial_y=2, serial_z=1
    )

    sch = at.tensorize_cuda(layer, tensorize_state, V100, tensorize_param)
    print(tvm.lower(sch, layer.schedule_tensors, simple_mode=True))


if __name__ == "__main__":
    test_build_fusion_choice()
    test_build_match_info()
    test_build_tensorize_hyper_fusion_state()
    test_tensorize_cuda()
