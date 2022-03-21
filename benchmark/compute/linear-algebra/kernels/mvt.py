import tvm


mvt_configs = {
    "mini": (40,),
    "small": (120,),
    "medium": (400,),
    "large": (2000,),
    "extra": (4000,),
}


def mvt(N):
    A = tvm.te.placeholder([N, N], name="A", dtype="float32")
    p = tvm.te.placeholder([N], name="p", dtype="float32")
    q = tvm.te.placeholder([N], name="q", dtype="float32")
    rn1 = tvm.te.reduce_axis([0, N], name="rn1")
    rn2 = tvm.te.reduce_axis([0, N], name="rn2")

    x = tvm.te.compute(
        [N], lambda n: tvm.te.sum(A[n, rn1] * p[rn1], axis=rn1), name="x"
    )

    y = tvm.te.compute(
        [N], lambda n: tvm.te.sum(A[rn2, n] * q[rn2], axis=rn2), name="y"
    )
    return [x, y], [A, p, q]


if __name__ == "__main__":
    (N,) = mvt_configs["large"]
    outs, ins = mvt(N)
    x, y = outs
    A, p, q = ins

    sch = tvm.te.create_schedule([x.op, y.op])
    print(tvm.lower(sch, [A, p, q, x, y], simple_mode=True))
