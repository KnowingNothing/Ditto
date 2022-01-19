import torch
import tvm
import numpy as np
import math
import argparse

MI = 16
NI = 16
KI = 16
WARP_SIZE = 32
IN_VEC = 4
OUT_VEC = 4


def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def BatchGemmSoftmaxGemm(batch, M, N, K, L, in_dtype="float16", acc_dtype="float32"):
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

    A_frag = tvm.te.compute(
        [batch, M // MI, K // KI, MI, KI],
        lambda b, mo, ko, mi, ki: A_shared[b, mo, ko, mi, ki],
        name="A_frag",
    )

    B_frag = tvm.te.compute(
        [batch, K // KI, L // NI, KI, NI],
        lambda b, ko, lo, ki, li: B_shared[b, ko, lo, ki, li],
        name="B_frag",
    )

    rko = tvm.te.reduce_axis([0, K // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A_frag[b, mo, rko, mi, rki].astype(acc_dtype)
            * B_frag[b, rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    D_shared = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: D_frag[b, mo, lo, mi, li],
        name="D_shared",
    )

    exp = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.exp(D_shared[b, mo, lo, mi, li]).astype(
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

    exp_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: exp[b, mo, lo, mi, li],
        name="exp_frag",
    )

    C_frag = tvm.te.compute(
        [batch, L // KI, ext_N // NI, KI, NI],
        lambda b, lo, no, li, ni: C_ext[b, lo, no, li, ni],
        name="C_frag",
    )

    rlo = tvm.te.reduce_axis([0, L // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, ext_N // NI, MI, NI],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            exp_frag[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_frag[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    E = tvm.te.compute(
        [batch, M // MI, ext_N // NI, MI, NI],
        lambda b, mo, no, mi, ni: E_frag[b, mo, no, mi, ni],
        name="E",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E[b, m // MI, n // NI, m % MI, n % NI].astype(in_dtype)
        / (
            E[b, m // MI, ext_N // NI - 1, m % MI, NI - 1].astype(in_dtype)
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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def schedule_llvm(batch, M, N, K, L, in_dtype="float16", acc_dtype="float32"):
    ins, outs = BatchGemmSoftmaxGemm(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )
    A, B, C = ins
    (F,) = outs

    sch = tvm.te.create_schedule(F.op)

    E = F.op.input_tensors[0]
    E_frag = E.op.input_tensors[0]
    exp_frag, C_frag = E_frag.op.input_tensors
    exp = exp_frag.op.input_tensors[0]
    D_shared = exp.op.input_tensors[0]
    C_ext = C_frag.op.input_tensors[0]
    D_frag = D_shared.op.input_tensors[0]
    A_frag, B_frag = D_frag.op.input_tensors
    A_shared = A_frag.op.input_tensors[0]
    B_shared = B_frag.op.input_tensors[0]

    sch[exp].compute_inline()
    sch[C_ext].compute_inline()
    sch[exp_frag].compute_inline()
    sch[C_frag].compute_inline()
    sch[A_shared].compute_inline()
    sch[B_shared].compute_inline()
    sch[A_frag].compute_inline()
    sch[B_frag].compute_inline()
    func = tvm.build(sch, [A, B, C, F], "llvm")
    return [A, B, C], [F], func


def schedule_cuda(batch, M, N, K, L, in_dtype="float16", acc_dtype="float32"):
    ins, outs = BatchGemmSoftmaxGemm(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )
    A, B, C = ins
    (F,) = outs

    sch = tvm.te.create_schedule(F.op)

    E = F.op.input_tensors[0]
    E_frag = E.op.input_tensors[0]
    exp_frag, C_frag = E_frag.op.input_tensors
    exp = exp_frag.op.input_tensors[0]
    D_shared = exp.op.input_tensors[0]
    C_ext = C_frag.op.input_tensors[0]
    D_frag = D_shared.op.input_tensors[0]
    A_frag, B_frag = D_frag.op.input_tensors
    A_shared = A_frag.op.input_tensors[0]
    B_shared = B_frag.op.input_tensors[0]

    sch[A_shared].set_scope("shared")
    sch[B_shared].set_scope("shared")
    sch[exp].set_scope("shared")
    exp_shared = exp
    sch[C_ext].set_scope("shared")
    C_ext_shared = C_ext
    sch[D_shared].set_scope("shared")

    sch[A_frag].set_scope("wmma.matrix_a")
    sch[B_frag].set_scope("wmma.matrix_b")
    sch[D_frag].set_scope("wmma.accumulator")
    sch[exp_frag].set_scope("wmma.matrix_a")
    sch[C_frag].set_scope("wmma.matrix_b")
    sch[E_frag].set_scope("wmma.accumulator")

    b, m, n = sch[F].op.axis
    fused = sch[F].fuse(b, m, n)
    bx, tx = sch[F].split(fused, factor=WARP_SIZE)
    sch[F].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    sch[F].bind(tx, tvm.te.thread_axis("threadIdx.x"))

    TZ_SIZE = 2
    TY_SIZE = 4
    UNROLL_STEP = 1500
    UNROLL_EXPLICIT = 1
    M_factors = [-1, TZ_SIZE, 1]
    N_factors = [-1, TY_SIZE, 2]
    L_K_factors = [-1, 1, TY_SIZE]
    K_factors = [-1, 16, 4]

    b, m, n, mi, ni = sch[E].op.axis
    m1, m2, m3 = tile_axes(sch, E, m, M_factors)
    n1, n2, n3 = tile_axes(sch, E, n, N_factors)
    sch[E].reorder(b, m1, n1, m2, n2, m3, n3, mi, ni)
    sch[E].bind(b, tvm.te.thread_axis("blockIdx.z"))
    sch[E].bind(m1, tvm.te.thread_axis("blockIdx.y"))
    sch[E].bind(n1, tvm.te.thread_axis("blockIdx.x"))
    sch[E].bind(m2, tvm.te.thread_axis("threadIdx.z"))
    sch[E].bind(n2, tvm.te.thread_axis("threadIdx.y"))
    sch[E].pragma(b, "auto_unroll_max_step", UNROLL_STEP)
    sch[E].pragma(b, "unroll_explicit", UNROLL_EXPLICIT)
    # tensorize mi
    sch[E].tensorize(mi, intrin_wmma_store_matrix("global", acc_dtype))
    E_frag_attach_tensor = E
    E_frag_attach_axis = n2

    sch[E_frag].compute_at(sch[E_frag_attach_tensor], E_frag_attach_axis)
    b, mo, no, mi, ni = sch[E_frag].op.axis
    (rl, rli) = sch[E_frag].op.reduce_axis
    rl1, rl2, rl3 = tile_axes(sch, E_frag, rl, L_K_factors)
    sch[E_frag].reorder(b, rl1, rl2, rl3, mo, no, mi, ni, rli)
    # tensorize mi
    sch[E_frag].tensorize(mi, intrin_wmma_gemm(in_dtype, acc_dtype))
    exp_frag_attach_tensor = E_frag
    C_ext_frag_attach_tensor = E_frag
    exp_shared_attach_tensor = E_frag
    C_ext_shared_attach_tensor = E_frag
    exp_frag_attach_axis = rl2
    C_ext_frag_attach_axis = rl2
    exp_shared_attach_axis = rl1
    C_ext_shared_attach_axis = rl1
    D_shared_attach_tensor = E_frag
    D_shared_attach_axis = rl1

    sch[exp_frag].compute_at(sch[exp_frag_attach_tensor], exp_frag_attach_axis)
    b, mo, lo, mi, li = sch[exp_frag].op.axis
    # sch[exp_frag].reorder(b, mo, lo, mi, li)
    # tensorize mi
    sch[exp_frag].tensorize(mi, intrin_wmma_load_matrix_a("float16"))

    sch[C_frag].compute_at(sch[C_ext_frag_attach_tensor], C_ext_frag_attach_axis)
    b, lo, no, li, ni = sch[C_frag].op.axis
    # sch[C_frag].reorder(b, lo, no, li, ni)
    # tensorize li
    sch[C_frag].tensorize(li, intrin_wmma_load_matrix_b("float16"))

    sch[exp_shared].compute_at(sch[exp_shared_attach_tensor], exp_shared_attach_axis)
    b, mo, lo, mi, li = sch[exp_shared].op.axis
    m1, m2 = sch[exp_shared].split(mo, nparts=TZ_SIZE)
    l1, l2 = sch[exp_shared].split(lo, nparts=TY_SIZE)
    sch[exp_shared].reorder(b, m1, l1, m2, l2, mi, li)
    sch[exp_shared].bind(m1, tvm.te.thread_axis("threadIdx.z"))
    sch[exp_shared].bind(l1, tvm.te.thread_axis("threadIdx.y"))

    sch[D_shared].compute_at(sch[D_shared_attach_tensor], D_shared_attach_axis)
    b, mo, lo, mi, li = sch[D_shared].op.axis
    m1, m2 = sch[D_shared].split(mo, nparts=TZ_SIZE)
    l1, l2 = sch[D_shared].split(lo, nparts=TY_SIZE)
    sch[D_shared].reorder(b, m1, l1, m2, l2, mi, li)
    sch[D_shared].bind(m1, tvm.te.thread_axis("threadIdx.z"))
    sch[D_shared].bind(l1, tvm.te.thread_axis("threadIdx.y"))
    # tensorize mi
    sch[D_shared].tensorize(mi, intrin_wmma_store_matrix("shared", acc_dtype))
    D_frag_attach_tensor = D_shared
    D_frag_attach_axis = l1

    sch[C_ext_shared].compute_at(
        sch[C_ext_shared_attach_tensor], C_ext_shared_attach_axis
    )
    b, l, n, li, ni = sch[C_ext_shared].op.axis
    fused = sch[C_ext_shared].fuse(b, l, n, li, ni)
    fused, tz, ty, tx, vec = tile_axes(
        sch, C_ext_shared, fused, [-1, TZ_SIZE, TY_SIZE, WARP_SIZE, IN_VEC]
    )
    sch[C_ext_shared].bind(tz, tvm.te.thread_axis("threadIdx.z"))
    sch[C_ext_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[C_ext_shared].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[C_ext_shared].vectorize(vec)

    sch[D_frag].compute_at(sch[D_frag_attach_tensor], D_frag_attach_axis)
    b, mo, lo, mi, li = sch[D_frag].op.axis
    (rk, rki) = sch[D_frag].op.reduce_axis
    rk1, rk2, rk3 = tile_axes(sch, D_frag, rk, K_factors)
    sch[D_frag].reorder(b, rk1, rk2, rk3, mo, lo, mi, li, rki)
    # tensorize mi
    sch[D_frag].tensorize(mi, intrin_wmma_gemm(in_dtype, acc_dtype))
    A_frag_attach_tensor = D_frag
    B_frag_attach_tensor = D_frag
    A_shared_attach_tensor = D_frag
    B_shared_attach_tensor = D_frag
    A_frag_attach_axis = rk2
    B_frag_attach_axis = rk2
    A_shared_attach_axis = rk1
    B_shared_attach_axis = rk1

    sch[A_frag].compute_at(sch[A_frag_attach_tensor], A_frag_attach_axis)
    b, mo, ko, mi, ki = sch[A_frag].op.axis
    sch[A_frag].reorder(b, mo, ko, mi, ki)
    # tensorize mi
    sch[A_frag].tensorize(mi, intrin_wmma_load_matrix_a("float16"))

    sch[B_frag].compute_at(sch[B_frag_attach_tensor], B_frag_attach_axis)
    b, ko, lo, ki, li = sch[B_frag].op.axis
    sch[B_frag].reorder(b, ko, lo, ki, li)
    # tensorize ki
    sch[B_frag].tensorize(ki, intrin_wmma_load_matrix_b("float16"))

    sch[A_shared].compute_at(sch[A_shared_attach_tensor], A_shared_attach_axis)
    b, m, k, mi, ki = sch[A_shared].op.axis
    fused = sch[A_shared].fuse(b, m, k, mi, ki)
    fused, tz, ty, tx, vec = tile_axes(
        sch, A_shared, fused, [-1, TZ_SIZE, TY_SIZE, WARP_SIZE, IN_VEC]
    )
    sch[A_shared].bind(tz, tvm.te.thread_axis("threadIdx.z"))
    sch[A_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[A_shared].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[A_shared].vectorize(vec)

    sch[B_shared].compute_at(sch[B_shared_attach_tensor], B_shared_attach_axis)
    b, k, l, ki, li = sch[B_shared].op.axis
    fused = sch[B_shared].fuse(b, k, l, ki, li)
    fused, tz, ty, tx, vec = tile_axes(
        sch, B_shared, fused, [-1, TZ_SIZE, TY_SIZE, WARP_SIZE, IN_VEC]
    )
    sch[B_shared].bind(tz, tvm.te.thread_axis("threadIdx.z"))
    sch[B_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[B_shared].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[B_shared].vectorize(vec)

    print(tvm.lower(sch, [A, B, C, F], simple_mode=True))
    func = tvm.build(sch, [A, B, C, F], "cuda")
    return [A, B, C], [F], func


def torch_bmm_softmax_bmm(A, B, C):
    D = torch.bmm(A, B)
    E = torch.softmax(D, dim=-1)
    F = torch.bmm(E, C)

    return F


def test_llvm():
    in_dtype = "float32"
    acc_dtype = "float32"
    ins, outs, func = schedule_llvm(
        12, 512, 64, 64, 512, in_dtype=in_dtype, acc_dtype=acc_dtype
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
        for y in outs
    ]
    ctx = tvm.cpu()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
    func(*inputs_tvm, *outputs_tvm)

    inputs_torch = [torch.tensor(x) for x in inputs_np]
    output = torch_bmm_softmax_bmm(*inputs_torch)
    from tvm import testing

    testing.assert_allclose(
        output.cpu().numpy(), outputs_tvm[0].asnumpy(), rtol=0.1, atol=0.1
    )


def test_cuda(profile):
    in_dtype = "float16"
    acc_dtype = "float32"
    ins, outs, func = schedule_cuda(
        12, 512, 64, 64, 512, in_dtype=in_dtype, acc_dtype=acc_dtype
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(in_dtype)
        for y in outs
    ]
    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
    if profile:
        func(*inputs_tvm, *outputs_tvm)
    else:
        # (TODO: size) I have checked the results, nan errors occurs
        # but currently I can't locate the cause of this error.

        evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
        cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        print(f"Our code uses {cost} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", choices=["correctness", "perf"])

    args = parser.parse_args()
    if args.mode == "correctness":
        test_llvm()
    elif args.mode == "perf":
        test_cuda(args.profile)
    else:
        raise ValueError()
