import tvm


def ceil(x, y):
    return (x + y - 1) // y


MI = 16
NI = 16
KI = 16


def conv_relu_conv_relu(N, C, H, W, K1, R1, S1, K2, R2, S2,
                        stride1=1, stride2=1, padding1=0, padding2=0,
                        in_dtype="float16", out_dtype="float32"):
    P1 = (H + 2 * padding1 - R1) // stride1 + 1
    Q1 = (W + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    CI = KI
    CO = ceil(C, KI)
    HWI = MI
    HWO = ceil(P1 * Q1, MI)
    K1I = NI
    K1O = ceil(K1, NI)
    PQI = MI
    PQO = ceil(P2 * Q2, MI)
    K2I = NI
    K2O = ceil(K2, NI)
    RK1I = KI
    RK1O = ceil(K1, KI)

    Img = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder(
        [K1, C, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder(
        [K2, K1, R2, S2], dtype=in_dtype, name="Weight2"
    )

    # shared scope begin
    pad1 = tvm.te.compute(
        [N, C, H + 2 * padding1, W + 2 * padding1],
        lambda n, c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding1, h < H + padding1,
                        w >= padding1, w < W + padding1),
                Img[n, c, h - padding1, w - padding1],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad1"
    )
    pad1_fact = tvm.te.compute(
        [N, CO, HWO, R1, S1, HWI, CI],
        lambda n, co, hwo, r1, s1, hwi, ci:
            tvm.tir.if_then_else(
                tvm.tir.all(co * CI + ci < C, (hwo * HWI + hwi) < P1*Q1),
                pad1[n, co * CI + ci, (hwo * HWI + hwi)//Q1 * stride1 +
                     r1, (hwo * HWI + hwi) % Q1 * stride1 + s1],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad1_fact"
    )
    Weight1_fact = tvm.te.compute(
        [K1O, CO, R1, S1, CI, K1I],
        lambda k1o, co, r1, s1, ci, k1i:
            tvm.tir.if_then_else(
                tvm.tir.all(k1o * K1I + k1i < K1, co * CI + ci < C),
                Weight1[k1o * K1I + k1i, co * CI + ci, r1, s1],
                tvm.tir.const(0, in_dtype)
        ),
        name="Weight1_fact"
    )
    # shared scope end

    # wmma begin
    pad1_fact_frag = tvm.te.compute(
        [N, CO, HWO, R1, S1, HWI, CI],
        lambda n, co, hwo, r1, s1, hwi, ci:
            pad1_fact[n, co, hwo, r1, s1, hwi, ci],
        name="pad1_fact_frag"
    )
    Weight1_fact_frag = tvm.te.compute(
        [K1O, CO, R1, S1, CI, K1I],
        lambda k1o, co, r1, s1, ci, k1i:
            Weight1_fact[k1o, co, r1, s1, ci, k1i],
        name="Weight1_fact_frag"
    )

    rc1o = tvm.te.reduce_axis([0, CO], "rc1o")
    rc1i = tvm.te.reduce_axis([0, CI], "rc1i")
    rr1 = tvm.te.reduce_axis([0, R1], "rr1")
    rs1 = tvm.te.reduce_axis([0, S1], "rs1")
    conv1_frag = tvm.te.compute(
        [N, K1O, HWO, HWI, K1I],
        lambda n, k1o, hwo, hwi, k1i:
            tvm.te.sum(
                pad1_fact_frag[n, rc1o, hwo, rr1,
                               rs1, hwi, rc1i] .astype(out_dtype)
                * Weight1_fact_frag[k1o, rc1o, rr1,
                                    rs1, rc1i, k1i].astype(out_dtype),
                axis=[rr1, rs1, rc1o, rc1i]
        ),
        name="conv1_frag"
    )

    conv1_shared = tvm.te.compute(
        [N, K1O, HWO, HWI, K1I],
        lambda n, k1o, hwo, kwi, k1i:
            conv1_frag[n, k1o, hwo, kwi, k1i],
        name="conv1_shared"
    )
    # wmma end

    # shared scope begin
    relu1 = tvm.te.compute(
        [N, K1O, HWO, HWI, K1I],
        lambda n, k1o, hwo, hwi, k1i:
            tvm.tir.if_then_else(
                conv1_shared[n, k1o, hwo, hwi,
                             k1i] > tvm.tir.const(0, out_dtype),
                conv1_shared[n, k1o, hwo, hwi, k1i].astype(in_dtype),
                tvm.tir.const(0, in_dtype)
        ),
        name="relu1"
    )

    if K1O * K1I > K1 or HWO * HWI > P1 * Q1:
        relu1_refact = tvm.te.compute(
            [N, K1, P1, Q1],
            lambda n, k1, p1, q1:
                relu1[n, k1//K1I, (p1 * Q1 + q1)//HWI,
                      (p1 * Q1 + q1) % HWI, k1 % K1I]
                + relu1[N-1, K1O-1, HWO-1, HWI-1, K1I-1],
            name="relu1_refact_trail"
        )
    else:
        relu1_refact = tvm.te.compute(
            [N, K1, P1, Q1],
            lambda n, k1, p1, q1:
                relu1[n, k1//K1I, (p1 * Q1 + q1)//HWI,
                      (p1 * Q1 + q1) % HWI, k1 % K1I],
            name="relu1_refact"
        )
    # shared scope end

    pad2 = tvm.te.compute(
        [N, K1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, k, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding2, h < P1 + padding2,
                            w >= padding2, w < Q1 + padding2),
                relu1_refact[n, k, h - padding2, w - padding2],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad2"
    )

    pad2_fact = tvm.te.compute(
        [N, RK1O, PQO, R2, S2, PQI, RK1I],
        lambda n, rk1o, pqo, r2, s2, pqi, rk1i:
            tvm.tir.if_then_else(
                tvm.tir.all(rk1o * RK1I + rk1i < K1,
                            (pqo * PQI + pqi) < P2 * Q2),
                pad2[n, rk1o * RK1I + rk1i, (pqo * PQI + pqi)//Q2 *
                     stride2 + r2, (pqo * PQI + pqi) % Q2 * stride2 + s2],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad2_fact"
    )

    Weigth2_fact = tvm.te.compute(
        [K2O, RK1O, R2, S2, RK1I, K2I],
        lambda k2o, rk1o, r2, s2, rk1i, k2i:
            tvm.tir.if_then_else(
                tvm.tir.all(k2o * K2I + k2i < K2, rk1o * RK1I + rk1i < K1),
                Weight2[k2o * K2I + k2i, rk1o * RK1I + rk1i < K1, r2, s2],
                tvm.tir.const(0, in_dtype)
        ),
        name="Weight2_fact"
    )

    # wmma begin
    pad2_fact_frag = tvm.te.compute(
        [N, RK1O, PQO, R2, S2, PQI, RK1I],
        lambda n, rk1o, pqo, r2, s2, pqi, rk1i:
            pad2_fact[n, rk1o, pqo, r2, s2, pqi, rk1i],
        name="pad2_fact_frag"
    )
    Weight2_fact_frag = tvm.te.compute(
        [K2O, RK1O, R2, S2, RK1I, K2I],
        lambda k2o, rk1o, r2, s2, rk1i, k2i:
            Weigth2_fact[k2o, rk1o, r2, s2, rk1i, k2i],
        name="Weight2_fact_frag"
    )
    rk1o = tvm.te.reduce_axis([0, RK1O], "rk1o")
    rk1i = tvm.te.reduce_axis([0, RK1I], "rk1i")
    rr2 = tvm.te.reduce_axis([0, R2], "rr2")
    rs2 = tvm.te.reduce_axis([0, S2], "rs2")
    conv2_frag = tvm.te.compute(
        [N, K2O, PQO, PQI, K2I],
        lambda n, k2o, pqo, pqi, k2i:
            tvm.te.sum(
                pad2_fact_frag[n, rk1o, pqo, rr2,
                               rs2, pqi, rk1i].astype(out_dtype)
                * Weight2_fact_frag[k2o, rk1o, rr2,
                                    rs2, rk1i, k2i].astype(out_dtype),
                axis=[rr2, rs2, rk1o, rk1i]
        ),
        name="conv2_frag"
    )

    conv2_shared = tvm.te.compute(
        [N, K2O, PQO, PQI, K2I],
        lambda n, k2o, pqo, pqi, k2i:
            conv2_frag[n, k2o, pqo, pqi, k2i],
        name="conv2_shared"
    )

    # wmma end

    # global scope begin
    relu2 = tvm.te.compute(
        [N, K2O, PQO, PQI, K2I],
        lambda n, k2o, pqo, pqi, k2i:
            tvm.tir.if_then_else(
                conv2_shared[n, k2o, pqo, pqi,
                             k2i] > tvm.tir.const(0, out_dtype),
                conv2_shared[n, k2o, pqo, pqi, k2i].astype(in_dtype),
                tvm.tir.const(0, in_dtype)
        ),
        name="relu2"
    )

    if K2O * K2I > K2 or PQO * PQI > P2 * Q2:
        relu2_refact = tvm.te.compute(
            [N, K2, P2, Q2],
            lambda n, k2, p2, q2:
                relu2[n, k2//K2I, (p2 * Q2 + q2)//PQI,
                      (p2 * Q2 + q2) % PQI, k2 % K2I]
                + relu2[N-1, K2O-1, PQO-1, PQI-1, K2I-1],
            name="relu2_refact_trail"
        )
    else:
        relu2_refact = tvm.te.compute(
            [N, K2, P2, Q2],
            lambda n, k2, p2, q2:
                relu2[n, k2//K2I, (p2 * Q2 + q2)//PQI,
                      (p2 * Q2 + q2) % PQI, k2 % K2I],
            name="relu2_refact"
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


def schedule_conv_relu_conv_relu(
    N, C, H, W, K1, R1, S1, K2, R2, S2,
    stride1=1, stride2=1, padding1=0, padding2=0,
    in_dtype="float16", out_dtype="float32"
):
    ins, outs = conv_relu_conv_relu(
        N, C, H, W, K1, R1, S1, K2, R2, S2,
        stride1=1, stride2=1, padding1=0, padding2=0,
        in_dtype="float32", out_dtype="float32"
    )
    Img, Weight1, Weight2 = ins
    relu2_refact, = outs
    
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
    # sch[pad1_fact_frag].set_scope("wmma.matrix_a")
    sch[pad1_fact_frag].set_scope("local")
    # sch[Weight1_fact_frag].set_scope("wmma.matrix_b")
    sch[Weight1_fact_frag].set_scope("local")
    # sch[conv1_frag].set_scope("wmma.accumulator")
    sch[conv1_frag].set_scope("local")
    sch[conv1_shared].set_scope("shared")
    sch[relu1].compute_inline()
    sch[relu1_refact].compute_inline()
    sch[pad2].compute_inline()
    sch[pad2_fact].set_scope("shared")
    sch[Weight2_fact].set_scope("shared")
    # sch[pad2_fact_frag].set_scope("wmma.matrix_a")
    # sch[Weight2_fact_frag].set_scope("wmma.matrix_b")
    # sch[conv2_frag].set_scope("wmma.accumulator")
    sch[pad2_fact_frag].set_scope("local")
    sch[Weight2_fact_frag].set_scope("local")
    sch[conv2_frag].set_scope("local")
    sch[conv2_shared].set_scope("shared")
    sch[relu2].compute_inline()
    
    WARP_SIZE = 32
    VEC_LEN = 1
    
    PQ_factors = [-1, 2, 1, MI]
    K2_factors = [-1, 4, 1, NI]
    TY_factor = PQ_factors[2] * K2_factors[2]
    RK1_factors = [-1, 4, 8]
    
    HW_factors = [-1, PQ_factors[2], 1, MI]
    K1_factors = [-1, K2_factors[2], 1, NI]
    RC_factors = [-1, 4, 2]
    
    n, k, p, q = sch[relu2_refact].op.axis
    pq = sch[relu2_refact].fuse(p, q)
    pq1, pq2, pq3, pqi = tile_axes(sch, relu2_refact, pq, PQ_factors)
    k1, k2, k3, ki = tile_axes(sch, relu2_refact, k, K2_factors)
    sch[relu2_refact].reorder(n, pq1, k1, pq2, k2, pq3, k3, pqi, ki)
    bx = sch[relu2_refact].fuse(n, pq1, k1)
    ty = sch[relu2_refact].fuse(pq2, k2)
    tx = sch[relu2_refact].fuse(pqi, ki)
    _, tx = sch[relu2_refact].split(tx, factor=WARP_SIZE)
    sch[relu2_refact].bind(bx, tvm.te.thread_axis("blockIdx.x"))
    sch[relu2_refact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[relu2_refact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    conv2_shared_attach_pos = bx
    
    sch[conv2_shared].compute_at(sch[relu2_refact], conv2_shared_attach_pos)
    n, ko, pqo, pqi, ki = sch[conv2_shared].op.axis
    pq2, pq3 = tile_axes(sch, conv2_shared, pqo, PQ_factors[1:-1])
    k2, k3 = tile_axes(sch, conv2_shared, ko, K2_factors[1:-1])
    sch[conv2_shared].reorder(n, pq2, k2, pq3, k3, pqi, ki)
    ty = sch[conv2_shared].fuse(pq2, k2)
    sch[conv2_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    # tensorize pqi
    conv2_frag_attach_pos = ty
    
    sch[conv2_frag].compute_at(sch[conv2_shared], conv2_frag_attach_pos)
    n, ko, pqo, pqi, ki = sch[conv2_frag].op.axis
    rr, rs, rco, rci = sch[conv2_frag].op.reduce_axis
    rco1, rco2, rco3 = tile_axes(sch, conv2_frag, rco, RK1_factors)
    sch[conv2_frag].reorder(n, rco1, rr, rs, rco2, ko, pqo, rco3, pqi, ki, rci)
    # tensorize pqi
    pad2_fact_frag_attach_pos = rco2
    weight2_fact_frag_attach_pos = rco2
    pad2_fact_attach_pos = rco1
    weight2_fact_attach_pos = rco1
    conv1_shared_attach_pos = rco1
    
    sch[pad2_fact_frag].compute_at(sch[conv2_frag], pad2_fact_frag_attach_pos)
    n, co, pqo, r, s, pqi, ci = sch[pad2_fact_frag].op.axis
    # tensorize pqi
    
    sch[Weight2_fact_frag].compute_at(sch[conv2_frag], weight2_fact_frag_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight2_fact_frag].op.axis
    # tensorize ci
    
    sch[pad2_fact].compute_at(sch[conv2_frag], pad2_fact_attach_pos)
    n, co, pqo, r, s, pqi, ci = sch[pad2_fact].op.axis
    fused = sch[pad2_fact].fuse(n, co, pqo, r, s, pqi, ci)
    fused, ty, tx, vec = tile_axes(sch, pad2_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN])
    sch[pad2_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[pad2_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[pad2_fact].vectorize(vec)
    
    sch[Weight2_fact].compute_at(sch[conv2_frag], weight2_fact_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight2_fact].op.axis
    fused = sch[Weight2_fact].fuse(ko, co, r, s, ci, ki)
    fused, ty, tx, vec = tile_axes(sch, Weight2_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN])
    sch[Weight2_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[Weight2_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[Weight2_fact].vectorize(vec)
    
    sch[conv1_shared].compute_at(sch[conv2_frag], conv1_shared_attach_pos)
    n, ko, hwo, hwi, ki = sch[conv1_shared].op.axis
    hwo1, hwo2, hwo3 = tile_axes(sch, conv1_shared, hwo, HW_factors[:-1])
    ko1, ko2, ko3 = tile_axes(sch, conv1_shared, ko, K1_factors[:-1])
    sch[conv1_shared].reorder(n, hwo1, ko1, hwo2, ko2, hwo3, ko3, hwi, ki)
    ty = sch[conv1_shared].fuse(hwo2, ko2)
    sch[conv1_shared].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    # tensorize hwi
    conv1_frag_attach_pos = ty
    
    sch[conv1_frag].compute_at(sch[conv1_shared], conv1_frag_attach_pos)
    n, ko, hwo, hwi, ki = sch[conv1_frag].op.axis
    rr, rs, rco, rci = sch[conv1_frag].op.reduce_axis
    rco1, rco2, rco3 = tile_axes(sch, conv1_frag, rco, RC_factors)
    sch[conv1_frag].reorder(n, rco1, rr, rs, rco2, ko, hwo, rco3, hwi, ki, rci)
    # tensorize hwi
    pad1_fact_frag_attach_pos = rco2
    weight1_fact_frag_attach_pos = rco2
    pad1_fact_attach_pos = rco1
    weight1_fact_attach_pos = rco1
    
    sch[pad1_fact_frag].compute_at(sch[conv1_frag], pad1_fact_frag_attach_pos)
    n, co, hwo, r, s, hwi, ci = sch[pad1_fact_frag].op.axis
    # tensorize hwi
    
    sch[Weight1_fact_frag].compute_at(sch[conv1_frag], weight1_fact_frag_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight1_fact_frag].op.axis
    # tensorize ci
    
    sch[pad1_fact].compute_at(sch[conv1_frag], pad1_fact_attach_pos)
    n, co, hwo, r, s, hwi, ci = sch[pad1_fact].op.axis
    fused = sch[pad1_fact].fuse(n, co, hwo, r, s, hwi, ci)
    fused, ty, tx, vec = tile_axes(sch, pad1_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN])
    sch[pad1_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[pad1_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[pad1_fact].vectorize(vec)
    
    sch[Weight1_fact].compute_at(sch[conv1_frag], weight1_fact_attach_pos)
    ko, co, r, s, ci, ki = sch[Weight1_fact].op.axis
    fused = sch[Weight1_fact].fuse(ko, co, r, s, ci, ki)
    fused, ty, tx, vec = tile_axes(sch, Weight1_fact, fused, [-1, TY_factor, WARP_SIZE, VEC_LEN])
    sch[Weight1_fact].bind(ty, tvm.te.thread_axis("threadIdx.y"))
    sch[Weight1_fact].bind(tx, tvm.te.thread_axis("threadIdx.x"))
    sch[Weight1_fact].vectorize(vec)
    
    print(tvm.lower(sch, [Img, Weight1, Weight2, relu2_refact], simple_mode=True))
    func = tvm.build(sch, [Img, Weight1, Weight2, relu2_refact], "cuda")
    
    
if __name__ == "__main__":
    schedule_conv_relu_conv_relu(
        1, 256, 32, 32, 512, 3, 3, 512, 1, 1,
        stride1=1, stride2=1, padding1=1, padding2=1,
        in_dtype="float16", out_dtype="float32"
    )

