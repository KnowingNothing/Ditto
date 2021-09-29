import tvm


_3mm_configs = {
    "mini": (16, 24, 22, 20, 18),
    "small": (40, 80, 70, 60, 50),
    "medium": (180, 220, 210, 200, 190),
    "large": (800, 1200, 1100, 1000, 900),
    "extra": (1600, 2400, 2200, 2000, 1800)
}


def _3mm(M, R, L, K, N):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, R], name="B", dtype="float32")
    C = tvm.te.placeholder([R, L], name="C", dtype="float32")
    D = tvm.te.placeholder([L, N], name="D", dtype="float32")
    
    rk = tvm.te.reduce_axis([0, K], name="rk")
    E = tvm.te.compute(
        [M, R],
        lambda m, r:
            tvm.te.sum(A[m, rk] * B[rk, r], axis=rk),
        name="E"
    )
    
    rl = tvm.te.reduce_axis([0, L], name="rl")
    F = tvm.te.compute(
        [R, N],
        lambda r, n:
            tvm.te.sum(C[r, rl] * D[rl, n], axis=rl),
        name="F"
    )
    
    rr = tvm.te.reduce_axis([0, R], name="rr")
    G = tvm.te.compute(
        [M, N],
        lambda m, n:
            tvm.te.sum(E[m, rr] + F[rr, n], axis=rr),
        name="G"
    )
    return [G], [A, B, C, D]


if __name__ == "__main__":
    M, R, L, K, N = _3mm_configs["large"]
    outs, ins = _3mm(M, R, L, K, N)
    G, = outs
    A, B, C, D = ins
    sch = tvm.te.create_schedule(G.op)
    print(tvm.lower(sch, [A, B, C, D, G], simple_mode=True))
    