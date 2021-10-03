import tvm


gemver_configs = {
    "mini": (40,),
    "small": (120,),
    "medium": (400,),
    "large": (2000,),
    "extra": (4000,)
}


def gemver(N):
    A = tvm.te.placeholder([N, N], name="A", dtype="float32")
    u1 = tvm.te.placeholder([N], name="u1", dtype="float32")
    u2 = tvm.te.placeholder([N], name="u2", dtype="float32")
    v1 = tvm.te.placeholder([N], name="v1", dtype="float32")
    v2 = tvm.te.placeholder([N], name="v2", dtype="float32")
    w = tvm.te.placeholder([N], name="w", dtype="float32")
    x = tvm.te.placeholder([N], name="x", dtype="float32")
    y = tvm.te.placeholder([N], name="y", dtype="float32")
    z = tvm.te.placeholder([N], name="z", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")
    
    B = tvm.te.compute(
        [N, N],
        lambda i, j: A[i, j] + u1[i] * v1[j] + u2[i] * v2[j], name="B")

    rn1 = tvm.te.reduce_axis([0, N], name="rn1")
    X = tvm.te.compute(
        [N],
        lambda n: tvm.te.sum(B[rn1, n] * y[rn1], axis=rn1),
        name="X"
    )

    XX = tvm.te.compute(
        [N],
        lambda n: x[n] + beta * X[n],
        name="XX"
    )
    
    XXX = tvm.te.compute(
        [N],
        lambda n: XX[n] + z[n],
        name="XXX"
    )
    
    rn2 = tvm.te.reduce_axis([0, N], name="rn2")
    W = tvm.te.compute(
        [N],
        lambda n: tvm.te.sum(B[n, rn2] * XXX[rn2], axis=rn2),
        name="W"
    )
    
    WW = tvm.te.compute(
        [N],
        lambda n: w[n] + alpha * W[n],
        name="WW"
    )
    return [WW], [A, u1, u2, v1, v2, w, x, y, z, alpha, beta]


if __name__ == "__main__":
    N, = gemver_configs["large"]
    outs, ins = gemver(N)
    WW, = outs
    A, u1, u2, v1, v2, w, x, y, z, alpha, beta = ins
    sch = tvm.te.create_schedule(WW.op)
    print(tvm.lower(sch, [A, u1, u2, v1, v2, w, x, y, z, alpha, beta, WW], simple_mode=True))
    