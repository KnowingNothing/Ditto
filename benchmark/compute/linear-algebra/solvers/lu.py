import tvm


lu_configs = {
    "mini": (40,),
    "small": (120,),
    "medium": (400,),
    "large": (2000,),
    "extra": (4000,),
}


def lu(N):
    A = tvm.te.placeholder([N, N], name="A", dtype="float32")
    state = tvm.te.placeholder([N, N, N], name="state", dtype="float32")

    init = tvm.te.compute([1, N, N], lambda _, i, j: A[i, j], name="init")

    rk = tvm.te.reduce_axis([0, N], name="rk")
    accum = tvm.te.compute(
        [N, N, N],
        lambda t, i, j: tvm.te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(i == t, j < i, rk < j),
                state[t - 1, i, rk] * state[t - 1, rk, j],
                tvm.tir.if_then_else(
                    tvm.tir.all(i == t, j >= i, rk < i),
                    state[t - 1, i, rk] * state[t - 1, rk, j],
                    0.0,
                ),
            ),
            axis=rk,
        ),
        name="accum",
    )

    sub = tvm.te.compute(
        [N, N, N], lambda t, i, j: state[t - 1, i, j] - accum[t, i, j], name="sub"
    )

    res = tvm.te.scan(init, sub, state)
    return [res], [A]


if __name__ == "__main__":
    (N,) = lu_configs["large"]
    outs, ins = lu(N)
    (res,) = outs
    (A,) = ins
    sch = tvm.te.create_schedule(res.op)
    print(tvm.lower(sch, [A, res], simple_mode=True))
