import tvm


symm_configs = {
    "mini": (20, 30),
    "small": (60, 80),
    "medium": (200, 240),
    "large": (1000, 1200),
    "extra": (2000, 2600)
}


def symm(M, N):
    A = tvm.te.placeholder([M, M], name="A", dtype="float32")
    B = tvm.te.placeholder([M, N], name="B", dtype="float32")
    C = tvm.te.placeholder([M, N], name="C", dtype="float32")
    state = tvm.te.placeholder([M, M, N], name="state", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")
    
    rm = tvm.te.reduce_axis([0, M], name="rm")
    tmp = tvm.te.compute(
        [M, N],
        lambda m, n:
            tvm.te.sum(A[m, rm] * B[rm, n], axis=rm),
        name="tmp"
    )
    
    D = tvm.te.compute(
        [M, N],
        lambda m, n:
            beta * C[m, n] + alpha * B[m, n] * A[m, m] + alpha * tmp[m, n],
        name="D"
    )
    
    init = tvm.te.compute(
        [1, M, N],
        lambda _, m, n:
            D[m, n],
        name="init"
    )
    
    update = tvm.te.compute(
        [M, M, N],
        lambda t, m, n:
            tvm.tir.if_then_else(
                m < t,
                state[t-1, m, n] + alpha * B[t, n] * A[t, m],
                state[t-1, m, n]
            ),
        name="update"
    )
    
    res = tvm.te.scan(init, update, state, inputs=[D, B,])
    return [res], [A, B, C, alpha, beta]


if __name__ == "__main__":
    M, N = symm_configs["large"]
    outs, ins = symm(M, N)
    res, = outs
    A, B, C, alpha, beta = ins
    sch = tvm.te.create_schedule(res.op)
    print(tvm.lower(sch, [A, B, C, alpha, beta, res], simple_mode=True))
    