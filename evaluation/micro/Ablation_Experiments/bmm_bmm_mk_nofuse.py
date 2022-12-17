import torch
import tvm
import numpy as np
import math
import argparse
import pickle as pkl

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


def BatchGemmGemm(batch, M, N, K, L, in_dtype="float16", acc_dtype="float32"):
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

    D = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: D_frag[b, mo, lo, mi, li],
        name="D",
    )

    D_shared = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: D[b, mo, mi, lo, li].astype(
            in_dtype
        ),
        name="D_shared",
    )

    C_shared = tvm.te.compute(
        [batch, L // KI, N // NI, KI, NI],
        lambda b, lo, no, li, ni: tvm.tir.if_then_else(
            no * NI + ni < N,
            C[b, lo * NI + li, no * NI + ni],
            tvm.tir.const(1, in_dtype),
        ),
        name="B_shared",
    )

    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: D_shared[b, mo, lo, mi, li],
        name="D_frag",
    )

    C_frag = tvm.te.compute(
        [batch, L // KI, N // NI, KI, NI],
        lambda b, lo, no, li, ni: C_shared[b, lo, no, li, ni],
        name="C_frag",
    )

    rlo = tvm.te.reduce_axis([0, L // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, N // NI, MI, NI],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            D_frag[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_frag[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    E = tvm.te.compute(
        [batch, M // MI, N // NI, MI, NI],
        lambda b, mo, no, mi, ni: E_frag[b, mo, no, mi, ni],
        name="E",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E[b, m // MI, n // NI, m % MI, n % NI].astype(in_dtype)
        / (
            E[b, m // MI, N // NI - 1, m % MI, NI - 1].astype(in_dtype)
            + tvm.tir.const(1e-5, in_dtype)
        ),
        name="F",
    )

    return [A, B, C], [F], D, E


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
    ins, outs = BatchGemmGemm(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )
    A, B, C = ins
    (F,) = outs

    sch = tvm.te.create_schedule(F.op)

    E = F.op.input_tensors[0]
    out_frag = E.op.input_tensors[0]
    A_frag, B_frag = out_frag.op.input_tensors
    A_shared = A_frag.op.input_tensors[0]
    D_global = A_shared.op.input_tensors[0]
    B_shared = B_frag.op.input_tensors[0]
    D_frag = D_global.op.input_tensors[0]
    A_frag, B_frag = D_frag.op.input_tensors
    A_shared = A_frag.op.input_tensors[0]
    B_shared = B_frag.op.input_tensors[0]

    sch[A_shared].compute_inline()
    sch[B_shared].compute_inline()
    sch[A_frag].compute_inline()
    sch[B_frag].compute_inline()
    sch[A_shared].compute_inline()
    sch[B_shared].compute_inline()
    sch[A_frag].compute_inline()
    sch[B_frag].compute_inline()
    func = tvm.build(sch, [A, B, C, F], "llvm")
    return [A, B, C], [D_global], func

def schedule_bmm(sch, out, in_dtype, acc_dtype):
    out_frag = out.op.input_tensors[0]
    A_frag, B_frag = out_frag.op.input_tensors
    A_shared = A_frag.op.input_tensors[0]
    B_shared = B_frag.op.input_tensors[0]
    
    sch[out].set_scope("global")
    sch[out_frag].set_scope("wmma.accumulator")
    sch[A_frag].set_scope("wmma.matrix_a")
    sch[B_frag].set_scope("wmma.matrix_b")
    sch[A_shared].set_scope("shared")
    sch[B_shared].set_scope("shared")

    TZ_SIZE = 2
    TY_SIZE = 4
    UNROLL_STEP = 512
    UNROLL_EXPLICIT = 1
    M_factors = [-1, TZ_SIZE, 1]
    N_factors = [-1, TY_SIZE, 2]
    L_K_factors = [-1, 1, TY_SIZE]

    b, m, n, mi, ni = sch[out].op.axis
    m1, m2, m3 = tile_axes(sch, out, m, M_factors)
    n1, n2, n3 = tile_axes(sch, out, n, N_factors)
    kernel_scope, b = sch[out].split(b, nparts=1)
    sch[out].pragma(kernel_scope, "auto_unroll_max_step", UNROLL_STEP)
    sch[out].pragma(kernel_scope, "unroll_explicit", UNROLL_EXPLICIT)
    sch[out].reorder(b, m1, n1, m2, n2, m3, n3, mi, ni)
    sch[out].bind(b, tvm.te.thread_axis("blockIdx.z"))
    sch[out].bind(m1, tvm.te.thread_axis("blockIdx.y"))
    sch[out].bind(n1, tvm.te.thread_axis("blockIdx.x"))
    sch[out].bind(m2, tvm.te.thread_axis("threadIdx.z"))
    sch[out].bind(n2, tvm.te.thread_axis("threadIdx.y"))
    # tensorize mi
    sch[out].tensorize(mi, intrin_wmma_store_matrix("global", acc_dtype))
    out_frag_attach_tensor = out
    out_frag_attach_axis = n2

    sch[out_frag].compute_at(sch[out_frag_attach_tensor], out_frag_attach_axis)
    b, mo, no, mi, ni = sch[out_frag].op.axis
    (rl, rli) = sch[out_frag].op.reduce_axis
    rl1, rl2, rl3 = tile_axes(sch, out_frag, rl, L_K_factors)
    sch[out_frag].reorder(b, rl1, rl2, rl3, mo, no, mi, ni, rli)
    # tensorize mi
    sch[out_frag].tensorize(mi, intrin_wmma_gemm(in_dtype, acc_dtype))
    A_frag_attach_tensor = out_frag
    B_frag_attach_tensor = out_frag
    exp_shared_attach_tensor = out_frag
    B_shared_attach_tensor = out_frag
    A_frag_attach_axis = rl2
    B_frag_attach_axis = rl2
    exp_shared_attach_axis = rl1
    B_shared_attach_axis = rl1
    A_shared_attach_tensor = out_frag
    A_shared_attach_axis = rl1

    sch[A_frag].compute_at(sch[A_frag_attach_tensor], A_frag_attach_axis)
    b, mo, lo, mi, li = sch[A_frag].op.axis
    sch[A_frag].tensorize(mi, intrin_wmma_load_matrix_a("float16"))

    sch[B_frag].compute_at(sch[B_frag_attach_tensor], B_frag_attach_axis)
    b, lo, no, li, ni = sch[B_frag].op.axis
    sch[B_frag].tensorize(li, intrin_wmma_load_matrix_b("float16"))

    sch[A_shared].compute_at(sch[A_shared_attach_tensor], A_shared_attach_axis)
    b, mo, lo, mi, li = sch[A_shared].op.axis
    sch[A_shared].reorder(b, mo, lo, mi, li)
    fused = sch[A_shared].fuse(mo, lo, mi, li)
    fused, tx, vec = tile_axes(sch, A_shared, fused, [-1, WARP_SIZE, 4])
    sch[A_shared].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[A_shared].vectorize(vec)

    sch[B_shared].compute_at(
        sch[B_shared_attach_tensor], B_shared_attach_axis
    )
    b, l, n, li, ni = sch[B_shared].op.axis
    fused = sch[B_shared].fuse(b, l, n, li, ni)
    fused, tz, ty, tx, vec = tile_axes(
        sch, B_shared, fused, [-1, TZ_SIZE, TY_SIZE, WARP_SIZE, IN_VEC]
    )
    sch[B_shared].bind(tz, tvm.te.thread_axis("threadIdx.z"))
    sch[B_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[B_shared].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[B_shared].vectorize(vec)

def schedule_cuda(batch, M, N, K, L, in_dtype="float16", acc_dtype="float32"):
    ins, outs, D, E = BatchGemmGemm(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )
    (F,) = outs

    sch = tvm.te.create_schedule(F.op)

    b, m, n = sch[F].op.axis
    fused = sch[F].fuse(b, m, n)
    bx, tx = sch[F].split(fused, factor=WARP_SIZE)
    sch[F].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    sch[F].bind(tx, tvm.te.thread_axis("threadIdx.x"))

    schedule_bmm(sch, E, in_dtype, acc_dtype)

    schedule_bmm(sch, D, in_dtype, acc_dtype)


    func = tvm.build(sch, ins + outs, "cuda")

    return ins, [F], func


def torch_bmm_softmax_bmm(A, B, C):
    D = torch.bmm(A, B)
    # E = torch.softmax(D, dim=-1)
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


def test_cuda(batch, M, N, K, L):
    in_dtype = "float16"
    acc_dtype = "float32"
    ins, outs, func = schedule_cuda(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in outs
    ]
    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=300)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    global time_cost 
    time_cost = cost 
    return cost

def getScore(shape):
    x = threading.Thread(target = test_cuda, args = shape)
    global time_cost
    time_cost = math.inf 
    x.start()
    x.join()
    return time_cost


def ceil(x, y):
    return (x + y - 1) // y

def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),  # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)),  # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)),  # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512),  # Mixer-Large/32-S
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "perf"])
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--record", action = "store_true"
    )
    args = parser.parse_args()
    if args.mode == "correctness":
        test_llvm()
    elif args.mode == "perf":
        costs = []
        for ss in shapes[args.begin : args.begin + args.num]:
            cost = test_cuda(*ss)
            costs.append((ss, cost))

        print("B,M,N,K,L")
        for cc in costs:
            print(
                f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{cc[1]}"
            )
        for cc in costs:
            print(cc[1])
        if args.record:
            with open("bmm_bmm_mk_nofuse.pkl", "wb") as f:
                pkl.dump(costs, f)
    else:
        raise ValueError()
