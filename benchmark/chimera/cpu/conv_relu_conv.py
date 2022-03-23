class MicroKernel:
    def __init__(self, N=1, K=4, H=3, W=8, C=1, R=3, S=3) -> None:
        self.N = N
        self.K = K
        self.H = H
        self.W = W
        self.C = C
        self.R = R
        self.S = S

    def RoundUp(self, N, K, H, W, C, R, S):
        def f(a, b):
            return (a + b - 1) // b * b

        return [
            f(N, self.N),
            f(K, self.K),
            f(H, self.H),
            f(W, self.W),
            f(C, self.C),
            f(R, self.R),
            f(S, self.S),
        ]

    def Verify(self, N, K, H, W, C, R, S):
        return (
            N % self.N == 0
            and K % self.K == 0
            and H % self.H == 0
            and W % self.W == 0
            and R % self.R == 0
            and S % self.S == 0
        )


# 1,128,388,388,64,3,3,64,3,3


def ConvReluConv(
    shape,
    padding1=1,
    padding2=1,
    stride1=1,
    stride2=1,
    in_dtype="float32",
    mk1=MicroKernel(),
    mk2=MicroKernel(),
):
    """
            N1 K4 H3 W8 C1 R3 S3
    Conv1   N C1 P0 Q0 C0 R1 S1
    Conv1   N C2 P1 Q1 C1 R2 S2
    """
    assert len(shape) == 10
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2 = shape
    P1 = (P0 + 2 * padding1 - R1) // stride1 + 1
    Q1 = (Q0 + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    assert C1 % mk1.K == 0
    assert C2 % mk2.K == 0
    assert P0 % mk1.H == 0
    assert P1 % mk2.H == 0
    assert Q0 % mk1.W == 0
    assert Q1 % mk2.W == 0
    assert C0 % mk1.C == 0
    assert C1 % mk2.C == 0

    Img = tvm.te.placeholder([N, C0, P0, Q0], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder([C1, C0, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([C2, C1, R2, S2], dtype=in_dtype, name="Weight2")

    Pad1 = tvm.te.compute(
        [N, C0, P0 + 2 * padding1, Q0 + 2 * padding1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding1, p < P0 + padding1, q >= padding1, q < Q0 + padding1
            ),
            Img[n, c, p - padding1, q - padding1],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad1",
    )

    r1 = tvm.te.reduce_axis((0, R1), name="rr1")
    s1 = tvm.te.reduce_axis((0, S1), name="rs1")
    co1 = tvm.te.reduce_axis((0, C0 // mk1.C), name="rco1")
    ci1 = tvm.te.reduce_axis((0, mk1.C), name="rci1")
    Conv1 = tvm.te.compute(
        [N // mk1.N, mk1.N, C1 // mk1.K, mk1.K, P1 // mk1.H, mk1.H, Q1 // mk1.W, mk1.W],
        lambda no1, ni1, ko1, ki1, ho1, hi1, wo1, wi1: tvm.te.sum(
            Pad1[
                no1 * mk1.N + ni1,
                co1 * mk1.C + ci1,
                ho1 * mk1.H + hi1 + r1,
                wo1 * mk1.W + wi1 + s1,
            ]
            * Weight1[ko1 * mk1.K + ki1, co1 * mk1.C + ci1, r1, s1],
            axis=[co1, ci1, r1, s1],
        ),
        name="conv1",
    )
    Conv1_rfact = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: Conv1[
            n // mk1.N,
            n % mk1.N,
            c // mk1.K,
            c % mk1.K,
            p // mk1.H,
            p % mk1.H,
            q // mk1.W,
            q % mk1.W,
        ],
        name="conv1_unfactored",
    )
    Conv1_relu = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            Conv1_rfact[n, c, p, q] > 0,
            Conv1_rfact[n, c, p, q],
            tvm.tir.const(0, in_dtype),
        ),
        name="relu",
    )

    Pad2 = tvm.te.compute(
        [N, C1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding2, p < P0 + padding2, q >= padding2, q < Q0 + padding2
            ),
            Conv1_relu[n, c, p - padding2, q - padding2],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad2",
    )

    r2 = tvm.te.reduce_axis((0, R2), name="rr2")
    s2 = tvm.te.reduce_axis((0, S2), name="rs2")
    co2 = tvm.te.reduce_axis((0, C1 // mk2.C), name="rco2")
    ci2 = tvm.te.reduce_axis((0, mk2.C), name="rci2")
    Conv2_fact = tvm.te.compute(
        [N // mk2.N, mk2.N, C2 // mk2.K, mk2.K, P2 // mk2.H, mk2.H, Q2 // mk2.W, mk2.W],
        lambda no2, ni2, ko2, ki2, ho2, hi2, wo2, wi2: tvm.te.sum(
            Pad2[
                no2 * mk2.N + ni2,
                co2 * mk2.C + ci2,
                ho2 * mk2.H + hi2 + r2,
                wo2 * mk2.W + wi2 + s2,
            ]
            * Weight2[ko2 * mk2.K + ki2, co2 * mk2.C + ci2, r2, s2],
            axis=[co2, ci2, r2, s2],
        ),
        name="conv2_fact",
    )
    Conv2 = tvm.te.compute(
        [N, C2, P2, Q2],
        lambda n, c, p, q: Conv2_fact[
            n // mk2.N,
            n % mk2.N,
            c // mk2.C,
            c % mk2.C,
            p // mk2.H,
            p % mk2.H,
            q // mk2.W,
            q % mk2.W,
        ],
        name="conv2",
    )
    return (
        [Img, Weight1, Weight2],
        [Conv2]
    )

