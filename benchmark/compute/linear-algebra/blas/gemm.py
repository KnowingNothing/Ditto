import tvm


gemm_configs = {
    "mini": (20, 25, 30),
    "small": (60, 70, 80),
    "medium": (200, 220, 240),
    "large": (1000, 1100, 1200),
    "extra": (2000, 2300, 2600),
}


def gemm(M, K, N):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, N], name="B", dtype="float32")
    C = tvm.te.placeholder([M, N], name="C", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")

    rk = tvm.te.reduce_axis([0, K], name="rk")
    tmp = tvm.te.compute(
        [M, N], lambda m, n: tvm.te.sum(A[m, rk] * B[rk, n], axis=rk), name="tmp"
    )

    D = tvm.te.compute(
        [M, N], lambda m, n: alpha * tmp[m, n] + beta * C[m, n], name="D"
    )
    return [D], [A, B, C, alpha, beta]


if __name__ == "__main__":
    M, N, K = gemm_configs["large"]
    outs, ins = gemm(M, K, N)
    (D,) = outs
    A, B, C, alpha, beta = ins
    sch = tvm.te.create_schedule(D.op)
    print(tvm.lower(sch, [A, B, C, alpha, beta, D], simple_mode=True))
