import tvm


def conv_relu_conv(N, C, H, W, K1, R1, S1, K2, R2, S2,
                   stride1=1, stride2=1, padding1=0, padding2=0,
                   in_dtype="float32", out_dtype="float32"):
    P1 = (H + 2 * padding1 - R1) // stride1
    Q1 = (W + 2 * padding1 - S1) // stride1
    P2 = (P1 + 2 * padding2 - R2) // stride2
    Q2 = (Q1 + 2 * padding2 - S2) // stride2
    Img = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder(
        [K1, C, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder(
        [K2, K1, R2, S2], dtype=in_dtype, name="Weight2"
    )
    pad1 = tvm.te.compute(
        [N, C, H + 2 * padding1, W + 2 * padding1],
        lambda n, c, h, w:
            tvm.if_then_else(
                tvm.all(h >= padding1, h < H + padding1,
                        w >= padding1, w < W + padding1),
                Img[n, c, h - padding1, w - padding1],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad1"
    )
    rc1 = tvm.te.reduce_axis([0, C], "rc1")
    rr1 = tvm.te.reduce_axis([0, R1], "rr1")
    rs1 = tvm.te.reduce_axis([0, S1], "rs1")
    conv1 = tvm.te.compute(
        [N, K1, P1, Q1],
        lambda n, k, p, q:
            tvm.te.sum(
                pad1[n, rc1, p + rr1, q +
                     rs1] .astype(out_dtype) * Weight1[k, rc1, rr1, rs1].astype(out_dtype),
                axis=[rc1, rr1, rs1]
        ),
        name="conv1"
    )
    relu = tvm.te.compute(
        [N, K1, P1, Q1],
        lambda n, k, p, q:
            tvm.tir.if_then_else(
                conv1[n, k, p, q] > tvm.tir.const(0, out_dtype),
                conv1[n, k, p, q].astype(in_dtype),
                tvm.tir.const(0, in_dtype)
        ),
        name="relu"
    )

    pad2 = tvm.te.compute(
        [N, K1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, k, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding2, h < P1 + padding2,
                            w >= padding2, w < Q1 + padding2),
                relu[n, k, h - padding2, w - padding2],
                tvm.tir.const(0, in_dtype)
        ),
        name="pad2"
    )

    rc2 = tvm.te.reduce_axis([0, K1], "rc2")
    rr2 = tvm.te.reduce_axis([0, R2], "rr2")
    rs2 = tvm.te.reduce_axis([0, S2], "rs2")
    conv2 = tvm.te.compute(
        [N, K2, P2, Q2],
        lambda n, k, p, q:
            tvm.te.sum(
                pad2[n, rc2, p + rr2, q +
                     rs2] .astype(out_dtype) * Weight2[k, rc2, rr2, rs2].astype(out_dtype),
                axis=[rc2, rr2, rs2]
        ),
        name="conv2"
    )

    return [Img, Weight1, Weight2], [conv2]


def schedule_conv3x3_relu_conv3x3(
    N, C, H, W, K1, R1, S1, K2, R2, S2,
    stride1=1, stride2=1, padding1=0, padding2=0,
    in_dtype="float32", out_dtype="float32"
):
    pass
