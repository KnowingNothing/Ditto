import tvm


bicg_configs = {
    "mini": (42, 38),
    "small": (124, 116),
    "medium": (410, 390),
    "large": (2100, 1900),
    "extra": (2200, 1800),
}


def bicg(M, N):
    A = tvm.te.placeholder([M, N], name="A", dtype="float32")
    p = tvm.te.placeholder([N], name="p", dtype="float32")
    q = tvm.te.placeholder([M], name="q", dtype="float32")
    rm = tvm.te.reduce_axis([0, M], name="rm")
    rn = tvm.te.reduce_axis([0, N], name="rn")

    x = tvm.te.compute([M], lambda m: tvm.te.sum(A[m, rn] * p[rn], axis=rn), name="x")

    y = tvm.te.compute([N], lambda n: tvm.te.sum(A[rm, n] * q[rm], axis=rm), name="y")
    return [x, y], [A, p, q]


if __name__ == "__main__":
    M, N = bicg_configs["large"]
    outs, ins = bicg(M, N)
    x, y = outs
    A, p, q = ins

    sch = tvm.te.create_schedule([x.op, y.op])
    print(tvm.lower(sch, [A, p, q, x, y], simple_mode=True))
