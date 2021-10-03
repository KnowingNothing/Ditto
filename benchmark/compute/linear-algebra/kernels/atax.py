import tvm

atax_configs = {
    "mini": (38, 42),
    "small": (116, 124),
    "medium": (390, 410),
    "large": (1900, 2100),
    "extra": (1800, 2200)
}


def atax(M, N):
    A = tvm.te.placeholder([M, N], name="A", dtype="float32")
    x = tvm.te.placeholder([N], name="x", dtype="float32")
    rk = tvm.te.reduce_axis([0, N], name="rk")
    tmp = tvm.te.compute([M], lambda i: tvm.te.sum(A[i, rk] * x[rk], axis=rk), name="tmp")
    rl = tvm.te.reduce_axis([0, M], name="rl")
    y = tvm.te.compute([N], lambda j: tvm.te.sum(A[rl, j] * tmp[rl], axis=rl), name="y")
    return [y,], [A, x]


if __name__ == "__main__":
    M, N = atax_configs["large"]
    outs, ins = atax(M, N)
    y, = outs
    A, x = ins
    
    sch = tvm.te.create_schedule(y.op)
    print(tvm.lower(sch, [A, x, y], simple_mode=True))