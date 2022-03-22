import tvm


doitgen_configs = {
    "mini": (8, 10, 12),
    "small": (20, 25, 30),
    "medium": (40, 50, 60),
    "large": (140, 150, 160),
    "extra": (220, 250, 270),
}


def doitgen(R, Q, P):
    A = tvm.te.placeholder([R, Q, P], name="A", dtype="float32")
    C4 = tvm.te.placeholder([P, P], name="C4", dtype="float32")

    rp = tvm.te.reduce_axis([0, P], name="rp")

    sum = tvm.te.compute(
        [R, Q, P],
        lambda r, q, p: tvm.te.sum(A[r, q, rp] * C4[rp, p], axis=rp),
        name="sum",
    )

    return [sum], [A, C4]


if __name__ == "__main__":
    print("[Warning] Implace update is not supported by TVM.")
    Q, R, P = doitgen_configs["large"]
    outs, ins = doitgen(R, Q, P)
    (res,) = outs
    A, C4 = ins

    sch = tvm.te.create_schedule([res.op])
    print(tvm.lower(sch, [A, C4, res], simple_mode=True))
