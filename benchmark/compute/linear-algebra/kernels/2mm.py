import tvm


_2mm_configs = {
    "mini": (16, 18, 22, 24),
    "small": (40, 50, 70, 80),
    "medium": (180, 190, 210, 220),
    "large": (800, 900, 1100, 1200),
    "extra": (1600, 1800, 2200, 2400),
}


def _2mm(M, L, K, N):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, L], name="B", dtype="float32")
    C = tvm.te.placeholder([L, N], name="C", dtype="float32")
    D = tvm.te.placeholder([M, N], name="D", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")

    rk = tvm.te.reduce_axis([0, K], name="rk")
    tmp = tvm.te.compute(
        [M, L], lambda m, l: tvm.te.sum(A[m, rk] * B[rk, l], axis=rk), name="tmp"
    )

    rl = tvm.te.reduce_axis([0, L], name="rl")
    E = tvm.te.compute(
        [M, N], lambda m, n: tvm.te.sum(tmp[m, rl] * C[rl, n], axis=rl), name="E"
    )
    F = tvm.te.compute([M, N], lambda m, n: alpha * E[m, n] + beta * D[m, n], name="F")
    return [F], [A, B, C, D, alpha, beta]


if __name__ == "__main__":
    M, L, K, N = _2mm_configs["large"]
    outs, ins = _2mm(M, L, K, N)
    (F,) = outs
    A, B, C, D, alpha, beta = ins
    sch = tvm.te.create_schedule(F.op)
    print(tvm.lower(sch, [A, B, C, D, alpha, beta, F], simple_mode=True))
