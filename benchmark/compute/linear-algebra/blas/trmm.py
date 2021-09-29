import tvm


trmm_configs = {
    "mini": (20, 30),
    "small": (60, 80),
    "medium": (200, 240),
    "large": (1000, 1200),
    "extra": (2000, 2600)
}


def trmm(M, N):
    A = tvm.te.placeholder([M, M], name="A", dtype="float32")
    B = tvm.te.placeholder([M, N], name="B", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    
    rm = tvm.te.reduce_axis([0, M], name="rm")
    tmp = tvm.te.compute(
        [M, N],
        lambda m, n:
            tvm.te.sum(
                tvm.tir.if_then_else(
                    rm > m,
                    A[rm, m] * B[rm, n],
                    0.0
                ), axis=rm),
        name="tmp"
    )
    
    C = tvm.te.compute(
        [M, N],
        lambda m, n:
            alpha * (B[m, n] + tmp[m, n]),
        name="C"
    )

    return [C], [A, B, alpha]


if __name__ == "__main__":
    M, N = trmm_configs["large"]
    outs, ins = trmm(M, N)
    C, = outs
    A, B, alpha = ins
    sch = tvm.te.create_schedule(C.op)
    print(tvm.lower(sch, [A, B, alpha, C], simple_mode=True))
    