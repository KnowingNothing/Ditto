import tvm


syrk_configs = {
    "mini": (20, 30),
    "small": (60, 80),
    "medium": (200, 240),
    "large": (1000, 1200),
    "extra": (2000, 2600),
}


def syrk(M, N):
    A = tvm.te.placeholder([N, M], name="A", dtype="float32")
    C = tvm.te.placeholder([N, N], name="C", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")

    D = tvm.te.compute(
        [N, N],
        lambda i, j: tvm.tir.if_then_else(j <= i, C[i, j] * beta, C[i, j]),
        name="D",
    )

    rm = tvm.te.reduce_axis([0, M], name="rm")
    tmp = tvm.te.compute(
        [N, N], lambda i, j: tvm.te.sum(A[j, rm] * A[i, rm], axis=rm), name="tmp"
    )

    E = tvm.te.compute(
        [N, N],
        lambda i, j: tvm.tir.if_then_else(j <= i, D[i, j] + alpha * tmp[i, j], D[i, j]),
        name="E",
    )

    return [E], [A, C, alpha, beta]


if __name__ == "__main__":
    M, N = syrk_configs["large"]
    outs, ins = syrk(M, N)
    (E,) = outs
    A, C, alpha, beta = ins
    sch = tvm.te.create_schedule(E.op)
    print(tvm.lower(sch, [A, C, alpha, beta, E], simple_mode=True))
