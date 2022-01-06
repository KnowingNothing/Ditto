import torch
import tvm
import numpy as np


def ceil(x, y):
    return (x + y - 1) // y


MI = 16
NI = 16
KI = 16
in_dtype = "float16"
acc_dtype = "float32"


def conv_relu_conv_relu(
    N,
    C,
    H,
    W,
    K1,
    R1,
    S1,
    K2,
    R2,
    S2,
    stride1=1,
    stride2=1,
    padding1=0,
    padding2=0,
    in_dtype="float16",
    acc_dtype="float32",
):
    Img = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder([K1, C, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([K2, K1, R2, S2], dtype=in_dtype, name="Weight2")
    P1 = (H + 2 * padding1 - R1) // stride1 + 1
    Q1 = (W + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    assert Q1 % MI == 0
    assert Q2 % MI == 0
    assert C % KI == 0
    assert K1 % NI == 0
    assert K1 % KI == 0
    assert K2 % NI == 0

    CI = KI
    CO = ceil(C, KI)
    Q1I = MI
    Q1O = ceil(Q1, MI)
    K1I = NI
    K1O = ceil(K1, NI)
    Q2I = MI
    Q2O = ceil(Q2, MI)
    K2I = NI
    K2O = ceil(K2, NI)
    RK1I = KI
    RK1O = ceil(K1, KI)

    

    # shared scope begin
    pad1 = tvm.te.compute(
        [N, C, H + 2 * padding1, W + 2 * padding1],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding1, h < H + padding1, w >= padding1, w < W + padding1
            ),
            Img[n, c, h - padding1, w - padding1],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad1",
    )
    pad1_fact = tvm.te.compute(
        [N, CO, R1, S1, P1, Q1O, Q1I, CI],
        lambda n, co, r, s, p, qo, qi, ci: tvm.tir.if_then_else(
            tvm.tir.all(co * CI + ci < C, (qo * Q1I + qi) < Q1),
            pad1[n, co * CI + ci, p * stride1 + r, (qo * Q1I + qi) * stride1 + s],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad1_fact",
    )
    Weight1_fact = tvm.te.compute(
        [K1O, CO, R1, S1, CI, K1I],
        lambda k1o, co, r1, s1, ci, k1i: tvm.tir.if_then_else(
            tvm.tir.all(k1o * K1I + k1i < K1, co * CI + ci < C),
            Weight1[k1o * K1I + k1i, co * CI + ci, r1, s1],
            tvm.tir.const(0, in_dtype),
        ),
        name="Weight1_fact",
    )
    # shared scope end

    # wmma begin
    pad1_fact_frag = tvm.te.compute(
        [N, CO, R1, S1, P1, Q1O, Q1I, CI],
        lambda n, co, r, s, p, qo, qi, ci: pad1_fact[n, co, r, s, p, qo, qi, ci],
        name="pad1_fact_frag",
    )
    Weight1_fact_frag = tvm.te.compute(
        [K1O, CO, R1, S1, CI, K1I],
        lambda k1o, co, r1, s1, ci, k1i: Weight1_fact[k1o, co, r1, s1, ci, k1i],
        name="Weight1_fact_frag",
    )

    rc1o = tvm.te.reduce_axis([0, CO], "rc1o")
    rc1i = tvm.te.reduce_axis([0, CI], "rc1i")
    rr1 = tvm.te.reduce_axis([0, R1], "rr1")
    rs1 = tvm.te.reduce_axis([0, S1], "rs1")
    conv1_frag = tvm.te.compute(
        [N, K1O, P1, Q1O, Q1I, K1I],
        lambda n, ko, p, qo, qi, ki: tvm.te.sum(
            pad1_fact_frag[n, rc1o, rr1, rs1, p, qo, qi, rc1i].astype(acc_dtype)
            * Weight1_fact_frag[ko, rc1o, rr1, rs1, rc1i, ki].astype(acc_dtype),
            axis=[rr1, rs1, rc1o, rc1i],
        ),
        name="conv1_frag",
    )

    conv1_shared = tvm.te.compute(
        [N, K1O, P1, Q1O, Q1I, K1I],
        lambda n, ko, p, qo, qi, ki: conv1_frag[n, ko, p, qo, qi, ki],
        name="conv1_shared",
    )
    # wmma end

    # shared scope begin
    relu1 = tvm.te.compute(
        [N, K1O, P1, Q1O, Q1I, K1I],
        lambda n, ko, p, qo, qi, ki: tvm.tir.if_then_else(
            conv1_shared[n, ko, p, qo, qi, ki] > tvm.tir.const(0, acc_dtype),
            conv1_shared[n, ko, p, qo, qi, ki].astype(in_dtype),
            tvm.tir.const(0, in_dtype),
        ),
        name="relu1",
    )

    if K1O * K1I > K1 or Q1O * Q1I > Q1:
        relu1_refact = tvm.te.compute(
            [N, K1, P1, Q1],
            lambda n, k1, p1, q1: relu1[n, k1 // K1I, p1, q1 // Q1I, q1 % Q1I, k1 % K1I]
            + relu1[N - 1, K1O - 1, P1 - 1, Q1O - 1, Q1I - 1, K1I - 1],
            name="relu1_refact_trail",
        )
    else:
        relu1_refact = tvm.te.compute(
            [N, K1, P1, Q1],
            lambda n, k1, p1, q1: relu1[
                n, k1 // K1I, p1, q1 // Q1I, q1 % Q1I, k1 % K1I
            ],
            name="relu1_refact",
        )
    # shared scope end

    pad2 = tvm.te.compute(
        [N, K1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, k, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding2, h < P1 + padding2, w >= padding2, w < Q1 + padding2
            ),
            relu1_refact[n, k, h - padding2, w - padding2],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad2",
    )

    pad2_fact = tvm.te.compute(
        [N, RK1O, R2, S2, P2, Q2O, Q2I, RK1I],
        lambda n, co, r, s, p, qo, qi, ci: tvm.tir.if_then_else(
            tvm.tir.all(co * RK1I + ci < K1, (qo * Q2I + qi) < Q2),
            pad2[n, co * RK1I + ci, p * stride2 + r, (qo * Q2I + qi) * stride2 + s],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad2_fact",
    )

    Weigth2_fact = tvm.te.compute(
        [K2O, RK1O, R2, S2, RK1I, K2I],
        lambda k2o, rk1o, r2, s2, rk1i, k2i: tvm.tir.if_then_else(
            tvm.tir.all(k2o * K2I + k2i < K2, rk1o * RK1I + rk1i < K1),
            Weight2[k2o * K2I + k2i, rk1o * RK1I + rk1i, r2, s2],
            tvm.tir.const(0, in_dtype),
        ),
        name="Weight2_fact",
    )

    # wmma begin
    pad2_fact_frag = tvm.te.compute(
        [N, RK1O, R2, S2, P2, Q2O, Q2I, RK1I],
        lambda n, co, r, s, p, qo, qi, ci: pad2_fact[n, co, r, s, p, qo, qi, ci],
        name="pad2_fact_frag",
    )
    Weight2_fact_frag = tvm.te.compute(
        [K2O, RK1O, R2, S2, RK1I, K2I],
        lambda k2o, rk1o, r2, s2, rk1i, k2i: Weigth2_fact[k2o, rk1o, r2, s2, rk1i, k2i],
        name="Weight2_fact_frag",
    )
    rk1o = tvm.te.reduce_axis([0, RK1O], "rk1o")
    rk1i = tvm.te.reduce_axis([0, RK1I], "rk1i")
    rr2 = tvm.te.reduce_axis([0, R2], "rr2")
    rs2 = tvm.te.reduce_axis([0, S2], "rs2")
    conv2_frag = tvm.te.compute(
        [N, K2O, P2, Q2O, Q2I, K2I],
        lambda n, ko, p, qo, qi, ki: tvm.te.sum(
            pad2_fact_frag[n, rk1o, rr2, rs2, p, qo, qi, rk1i].astype(acc_dtype)
            * Weight2_fact_frag[ko, rk1o, rr2, rs2, rk1i, ki].astype(acc_dtype),
            axis=[rr2, rs2, rk1o, rk1i],
        ),
        name="conv2_frag",
    )

    conv2_shared = tvm.te.compute(
        [N, K2O, P2, Q2O, Q2I, K2I],
        lambda n, ko, p, qo, qi, ki: conv2_frag[n, ko, p, qo, qi, ki],
        name="conv2_shared",
    )

    # wmma end

    # global scope begin
    relu2 = tvm.te.compute(
        [N, K2O, P2, Q2O, Q2I, K2I],
        lambda n, ko, p, qo, qi, ki: tvm.tir.if_then_else(
            conv2_shared[n, ko, p, qo, qi, ki] > tvm.tir.const(0, acc_dtype),
            conv2_shared[n, ko, p, qo, qi, ki].astype(in_dtype),
            tvm.tir.const(0, in_dtype),
        ),
        name="relu2",
    )

    if K2O * K2I > K2 or Q2O * Q2I > Q2:
        relu2_refact = tvm.te.compute(
            [N, K2, P2, Q2],
            lambda n, k, p, q: relu2[n, k // K2I, p, q // Q2I, q % Q2I, k % K2I]
            + relu2[N - 1, K2O - 1, P2 - 1, Q2O - 1, Q2I - 1, K2I - 1],
            name="relu2_refact_trail",
        )
    else:
        relu2_refact = tvm.te.compute(
            [N, K2, P2, Q2],
            lambda n, k, p, q: relu2[n, k // K2I, p, q // Q2I, q % Q2I, k % K2I],
            name="relu2_refact",
        )
    # global scope end

    return [Img, Weight1, Weight2], [relu2_refact]


def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def intrin_wmma_load_matrix_a():
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


def intrin_wmma_load_matrix_b():
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


def intrin_wmma_gemm():
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


def intrin_wmma_store_matrix(scope):
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


def schedule_conv_relu_conv_relu(
    N,
    C,
    H,
    W,
    K1,
    R1,
    S1,
    K2,
    R2,
    S2,
    stride1=1,
    stride2=1,
    padding1=0,
    padding2=0,
    in_dtype="float16",
    acc_dtype="float32",
):
    ins, outs = conv_relu_conv_relu(
        N,
        C,
        H,
        W,
        K1,
        R1,
        S1,
        K2,
        R2,
        S2,
        stride1=stride1,
        stride2=stride2,
        padding1=padding1,
        padding2=padding2,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
    )
    Img, Weight1, Weight2 = ins
    (relu2_refact,) = outs

    sch = tvm.te.create_schedule(relu2_refact.op)
    relu2 = relu2_refact.op.input_tensors[0]
    conv2_shared = relu2.op.input_tensors[0]
    conv2_frag = conv2_shared.op.input_tensors[0]
    pad2_fact_frag = conv2_frag.op.input_tensors[0]
    Weight2_fact_frag = conv2_frag.op.input_tensors[1]
    pad2_fact = pad2_fact_frag.op.input_tensors[0]
    Weight2_fact = Weight2_fact_frag.op.input_tensors[0]
    pad2 = pad2_fact.op.input_tensors[0]
    relu1_refact = pad2.op.input_tensors[0]
    relu1 = relu1_refact.op.input_tensors[0]
    conv1_shared = relu1.op.input_tensors[0]
    conv1_frag = conv1_shared.op.input_tensors[0]
    pad1_fact_frag = conv1_frag.op.input_tensors[0]
    Weight1_fact_frag = conv1_frag.op.input_tensors[1]
    pad1_fact = pad1_fact_frag.op.input_tensors[0]
    Weight1_fact = Weight1_fact_frag.op.input_tensors[0]
    pad1 = pad1_fact.op.input_tensors[0]

    sch[pad1].compute_inline()
    sch[pad1_fact].set_scope("shared")
    sch[Weight1_fact].set_scope("shared")
    sch[pad1_fact_frag].set_scope("wmma.matrix_a")
    sch[Weight1_fact_frag].set_scope("wmma.matrix_b")
    sch[conv1_frag].set_scope("wmma.accumulator")
    # sch[pad1_fact_frag].set_scope("local")
    # sch[Weight1_fact_frag].set_scope("local")
    # sch[conv1_frag].set_scope("local")
    sch[conv1_shared].set_scope("shared")
    sch[relu1].compute_inline()
    sch[relu1_refact].compute_inline()
    sch[pad2].compute_inline()
    sch[pad2_fact].set_scope("shared")
    sch[Weight2_fact].set_scope("shared")
    sch[pad2_fact_frag].set_scope("wmma.matrix_a")
    sch[Weight2_fact_frag].set_scope("wmma.matrix_b")
    sch[conv2_frag].set_scope("wmma.accumulator")
    # sch[pad2_fact_frag].set_scope("local")
    # sch[Weight2_fact_frag].set_scope("local")
    # sch[conv2_frag].set_scope("local")
    sch[conv2_shared].set_scope("shared")
    sch[relu2].compute_inline()

    WARP_SIZE = 32
    VEC_LEN = 4

    P2_factors = [-1, 1, 1]
    Q2_factors = [-1, 1, 1, MI]
    K2_factors = [-1, 1, 1, NI]
    TY_factor = P2_factors[1] * Q2_factors[1] * K2_factors[1]
    RK1_factors = [-1, 1, 1]

    P1_factors = [-1, P2_factors[1], 1]
    Q1_factors = [-1, Q2_factors[1], 1, MI]
    K1_factors = [-1, K2_factors[2], 1, NI]
    RC_factors = [-1, 1, 2]

    n, k, p, q = sch[relu2_refact].op.axis
    q1, q2, q3, qi = tile_axes(sch, relu2_refact, q, Q2_factors)
    p1, p2, p3 = tile_axes(sch, relu2_refact, p, P2_factors)
    k1, k2, k3, ki = tile_axes(sch, relu2_refact, k, K2_factors)
    sch[relu2_refact].reorder(n, k1, p1, q1, k2, p2, q2, k3, p3, q3, qi, ki)
    bx = sch[relu2_refact].fuse(n, k1, p1, q1)
    ty = sch[relu2_refact].fuse(k2, p2, q2)
    tx = sch[relu2_refact].fuse(qi, ki)
    _, tx = sch[relu2_refact].split(tx, factor=WARP_SIZE)
    sch[relu2_refact].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    sch[relu2_refact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[relu2_refact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    conv2_shared_attach_pos = bx

    sch[conv2_shared].compute_at(sch[relu2_refact], conv2_shared_attach_pos)
    n, ko, p, qo, qi, ki = sch[conv2_shared].op.axis
    q1, q2, q3 = tile_axes(sch, conv2_shared, qo, Q2_factors[:-1])
    p1, p2, p3 = tile_axes(sch, conv2_shared, p, P2_factors)
    k1, k2, k3 = tile_axes(sch, conv2_shared, ko, K2_factors[:-1])
    sch[conv2_shared].reorder(n, k1, p1, q1, k2, p2, q2, k3, p3, q3, qi, ki)
    ty = sch[conv2_shared].fuse(k2, p2, q2)
    sch[conv2_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    # tensorize qi
    sch[conv2_shared].tensorize(qi, intrin_wmma_store_matrix("shared"))
    conv2_frag_attach_pos = ty

    sch[conv2_frag].compute_at(sch[conv2_shared], conv2_frag_attach_pos)
    n, ko, p, qo, qi, ki = sch[conv2_frag].op.axis
    rr, rs, rco, rci = sch[conv2_frag].op.reduce_axis
    rco1, rco2, rco3 = tile_axes(sch, conv2_frag, rco, RK1_factors)
    sch[conv2_frag].reorder(n, rco1, rr, rs, rco2, ko, p, qo, rco3, qi, ki, rci)
    # tensorize qi
    sch[conv2_frag].tensorize(qi, intrin_wmma_gemm())
    pad2_fact_frag_attach_pos = rco2
    weight2_fact_frag_attach_pos = rco2
    pad2_fact_attach_pos = rco1
    weight2_fact_attach_pos = rco1
    conv1_shared_attach_pos = rco1

    sch[pad2_fact_frag].compute_at(sch[conv2_frag], pad2_fact_frag_attach_pos)
    n, co, r, s, p, qo, qi, ci = sch[pad2_fact_frag].op.axis
    # tensorize qi
    sch[pad2_fact_frag].tensorize(qi, intrin_wmma_load_matrix_a())

    sch[Weight2_fact_frag].compute_at(sch[conv2_frag], weight2_fact_frag_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight2_fact_frag].op.axis
    # tensorize ci
    sch[Weight2_fact_frag].tensorize(ci, intrin_wmma_load_matrix_b())

    sch[pad2_fact].compute_at(sch[conv2_frag], pad2_fact_attach_pos)
    n, co, r, s, p, qo, qi, ci = sch[pad2_fact].op.axis
    fused = sch[pad2_fact].fuse(n, co, r, s, p, qo, qi, ci)
    fused, ty, tx, vec = tile_axes(
        sch, pad2_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN]
    )
    sch[pad2_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[pad2_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[pad2_fact].vectorize(vec)

    sch[Weight2_fact].compute_at(sch[conv2_frag], weight2_fact_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight2_fact].op.axis
    fused = sch[Weight2_fact].fuse(ko, co, r, s, ci, ki)
    fused, ty, tx, vec = tile_axes(
        sch, Weight2_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN]
    )
    sch[Weight2_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[Weight2_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[Weight2_fact].vectorize(vec)

    sch[conv1_shared].compute_at(sch[conv2_frag], conv1_shared_attach_pos)
    n, ko, p, qo, qi, ki = sch[conv1_shared].op.axis
    q1, q2, q3 = tile_axes(sch, conv1_shared, qo, Q1_factors[:-1])
    p1, p2, p3 = tile_axes(sch, conv1_shared, p, P1_factors)
    k1, k2, k3 = tile_axes(sch, conv1_shared, ko, K1_factors[:-1])
    sch[conv1_shared].reorder(n, k1, p1, q1, k2, p2, q2, k3, p3, q3, qi, ki)
    ty = sch[conv1_shared].fuse(k2, p2, q2)
    sch[conv1_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    # tensorize qi
    sch[conv1_shared].tensorize(qi, intrin_wmma_store_matrix("shared"))
    conv1_frag_attach_pos = ty

    sch[conv1_frag].compute_at(sch[conv1_shared], conv1_frag_attach_pos)
    n, ko, p, qo, qi, ki = sch[conv1_frag].op.axis
    rr, rs, rco, rci = sch[conv1_frag].op.reduce_axis
    rco1, rco2, rco3 = tile_axes(sch, conv1_frag, rco, RC_factors)
    sch[conv1_frag].reorder(n, rco1, rr, rs, rco2, ko, p, qo, rco3, qi, ki, rci)
    # tensorize qi
    sch[conv1_frag].tensorize(qi, intrin_wmma_gemm())
    pad1_fact_frag_attach_pos = rco2
    weight1_fact_frag_attach_pos = rco2
    pad1_fact_attach_pos = rco1
    weight1_fact_attach_pos = rco1

    sch[pad1_fact_frag].compute_at(sch[conv1_frag], pad1_fact_frag_attach_pos)
    n, co, r, s, p, qo, qi, ci = sch[pad1_fact_frag].op.axis
    # tensorize qi
    sch[pad1_fact_frag].tensorize(qi, intrin_wmma_load_matrix_a())

    sch[Weight1_fact_frag].compute_at(sch[conv1_frag], weight1_fact_frag_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight1_fact_frag].op.axis
    # tensorize ci
    sch[Weight1_fact_frag].tensorize(ci, intrin_wmma_load_matrix_b())

    sch[pad1_fact].compute_at(sch[conv1_frag], pad1_fact_attach_pos)
    n, co, r, s, p, qo, qi, ci = sch[pad1_fact].op.axis
    fused = sch[pad1_fact].fuse(n, co, r, s, p, qo, qi, ci)
    fused, ty, tx, vec = tile_axes(
        sch, pad1_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN]
    )
    sch[pad1_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[pad1_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[pad1_fact].vectorize(vec)

    sch[Weight1_fact].compute_at(sch[conv1_frag], weight1_fact_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight1_fact].op.axis
    fused = sch[Weight1_fact].fuse(ko, co, r, s, ci, ki)
    fused, ty, tx, vec = tile_axes(
        sch, Weight1_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN]
    )
    sch[Weight1_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[Weight1_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[Weight1_fact].vectorize(vec)

    print(tvm.lower(sch, [Img, Weight1, Weight2, relu2_refact], simple_mode=True))
    func = tvm.build(sch, [Img, Weight1, Weight2, relu2_refact], "cuda")
    return [Img, Weight1, Weight2], [relu2_refact], func


def torch_conv_relu_conv_relu(
    Img, Weigh1, Weight2, stride1=1, stride2=1, padding1=0, padding2=0
):
    conv1 = torch.nn.functional.conv2d(
        Img, Weigh1, bias=None, stride=stride1, padding=padding1
    )
    relu1 = torch.relu(conv1)
    conv2 = torch.nn.functional.conv2d(
        relu1, Weight2, bias=None, stride=stride2, padding=padding2
    )
    relu2 = torch.relu(conv2)
    return relu2


def schedule_conv_relu_conv_relu_cpu(
    N,
    C,
    H,
    W,
    K1,
    R1,
    S1,
    K2,
    R2,
    S2,
    stride1=1,
    stride2=1,
    padding1=0,
    padding2=0,
    in_dtype="float16",
    acc_dtype="float32",
):
    ins, outs = conv_relu_conv_relu(
        N,
        C,
        H,
        W,
        K1,
        R1,
        S1,
        K2,
        R2,
        S2,
        stride1=stride1,
        stride2=stride2,
        padding1=padding1,
        padding2=padding2,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
    )
    Img, Weight1, Weight2 = ins
    (relu2_refact,) = outs

    sch = tvm.te.create_schedule(relu2_refact.op)
    func = tvm.build(sch, [Img, Weight1, Weight2, relu2_refact])
    return [Img, Weight1, Weight2], [relu2_refact], func


def test_cuda():
    ins, outs, func = schedule_conv_relu_conv_relu(
        16,
        256,
        32,
        16,
        512,
        3,
        3,
        512,
        3,
        3,
        stride1=1,
        stride2=1,
        padding1=1,
        padding2=1,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
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
    func(*inputs_tvm, *outputs_tvm)

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    output = torch_conv_relu_conv_relu(
        *inputs_torch, stride1=1, stride2=1, padding1=1, padding2=1
    )
    from tvm import testing

    testing.assert_allclose(
        output.cpu().numpy(), outputs_tvm[0].asnumpy(), rtol=0.1, atol=0.1
    )
    
    
def test_llvm():
    ins, outs, func = schedule_conv_relu_conv_relu_cpu(
        1,
        256,
        32,
        32,
        512,
        3,
        3,
        512,
        3,
        3,
        stride1=1,
        stride2=1,
        padding1=1,
        padding2=1,
        in_dtype=in_dtype,
        acc_dtype=acc_dtype,
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

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    output = torch_conv_relu_conv_relu(
        *inputs_torch, stride1=1, stride2=1, padding1=1, padding2=1
    )
    from tvm import testing

    testing.assert_allclose(
        output.cpu().numpy(), outputs_tvm[0].asnumpy(), rtol=0.1, atol=0.1
    )


if __name__ == "__main__":
    test_cuda()
