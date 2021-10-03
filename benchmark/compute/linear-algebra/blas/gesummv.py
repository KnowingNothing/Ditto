import tvm


gesummv_configs = {
    "mini": (30,),
    "small": (90,),
    "medium": (250,),
    "large": (1300,),
    "extra": (2000,)
}


def gesummv(N):
    A = tvm.te.placeholder([N, N], name="A", dtype="float32")
    B = tvm.te.placeholder([N, N], name="B", dtype="float32")
    x = tvm.te.placeholder([N], name="x", dtype="float32")
    alpha = tvm.tir.Var("alpha", "float32")
    beta = tvm.tir.Var("beta", "float32")
    
    rn1 = tvm.te.reduce_axis([0, N], name="rn1")
    left = tvm.te.compute(
        [N],
        lambda n:
            tvm.te.sum(A[n, rn1] * x[rn1], axis=rn1),
        name="left"
    )
    
    rn2 = tvm.te.reduce_axis([0, N], name="rn2")
    right = tvm.te.compute(
        [N],
        lambda n:
            tvm.te.sum(B[n, rn2] * x[rn2], axis=rn2),
        name="right"
    )
    
    y = tvm.te.compute(
        [N],
        lambda n:
            alpha * left[n] + beta * right[n],
        name="y"
    )
    return [y], [A, B, x, alpha, beta]


if __name__ == "__main__":
    N, = gesummv_configs["large"]
    outs, ins = gesummv(N)
    y, = outs
    A, B, x, alpha, beta = ins
    sch = tvm.te.create_schedule(y.op)
    print(tvm.lower(sch, [A, B, x, alpha, beta, y], simple_mode=True))
    