import tvm 
import numpy as np
import argparse
import random
import pickle as pkl
import math
import threading

TY = 8
TX = 8
TL = 32
TK = 8
WARP_Y = 32
WARP_X = 1
VEC = 1
UNROLL_STEP = 512
UNROLL_EXPLICIT = 1

def tile_axes(s, axis, factors):
    ret = []
    for f in reversed(factors):
        axis, inner = s.split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))

def bind(s, binds):
    for (ax, thread_ax) in binds:
        s.bind(ax, tvm.te.thread_axis(thread_ax))

def schedule_bmm(sch, A, B, C, fusion = True, second = False):
    A_shared = A if (fusion and second) else sch.cache_read(A, "shared", [C])
    B_shared = sch.cache_read(B, "shared", [C])
    A_local = sch.cache_read(A_shared, "local", [C])
    B_local = sch.cache_read(B_shared, "local", [C])
    C_local = sch.cache_write(C, "local")

    b, m, n = sch[C].op.axis 
    mo, mm, mi = tile_axes(sch[C], m, (TY, WARP_Y))
    no, nm, ni = tile_axes(sch[C], n, (TX, WARP_X))

    sch[C].reorder(b, mo, no, mm, nm, mi, ni)
    sch[C].pragma(b, "auto_unroll_max_step", UNROLL_STEP)
    sch[C].pragma(b, "unroll_explicit", UNROLL_EXPLICIT)
    bind(sch[C], [(b, "blockIdx.z"), (mo, "blockIdx.y"), (no, "blockIdx.x"), 
        (mm, "threadIdx.y"), (nm, "threadIdx.x")])

    sch[C_local].compute_at(sch[C], nm)
    b,m,n = sch[C_local].op.axis
    (rk,) = sch[C_local].op.reduce_axis 
    rko, rki = sch[C_local].split(rk, TL)
    sch[C_local].reorder(b, rko, rki, m, n)

    sch[A_local].compute_at(sch[C_local], rki)
    sch[B_local].compute_at(sch[C_local], rki)

    sch[A_shared].compute_at(sch[C_local], rko)
    if not (fusion and second):
        b,m,k = sch[A_shared].op.axis 
        fused = sch[A_shared].fuse(b, m, k)
        fused, ty, tx, vec = tile_axes(sch[A_shared], fused, (TY, TX, VEC))
        bind(sch[A_shared], [(ty, "threadIdx.y"), (tx, "threadIdx.x")])
        sch[A_shared].vectorize(vec)

    sch[B_shared].compute_at(sch[C_local], rko)
    b, k, n = sch[B_shared].op.axis 
    fused = sch[B_shared].fuse(b, k, n)
    fused, ty, tx, vec = tile_axes(sch[B_shared], fused, (TY, TX, VEC))
    bind(sch[B_shared], [(ty, "threadIdx.y"), (tx, "threadIdx.x")])
    sch[B_shared].vectorize(vec)
    return C_local, rko

def BatchGemmGemm(batch, M, N, K, L, in_dtype, acc_dtype):
    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=in_dtype)

    rk = tvm.te.reduce_axis([0, K], "rk")
    D = tvm.te.compute([batch, M, L], 
        lambda b, m, l: tvm.te.sum(
            A[b, m, rk] * B[b, rk, l], 
            axis = rk
        ),
        name = "D"
    )

    rl = tvm.te.reduce_axis([0, L], "rl")
    E = tvm.te.compute(
        [batch, M, N], 
        lambda b, m, n: tvm.te.sum(
            C[b, m, rl] * D[b, rl, n],
            axis = rl
        ),
        name = "E"
    )
    return A, B, C, D, E

def schedule_cuda(batch, M, N, K, L, in_dtype = "float16", acc_dtype = "float32"):
    A, B, C, D, E = BatchGemmGemm(batch, M, N, K, L, in_dtype, acc_dtype)
    sch = tvm.te.create_schedule(E.op)
    # print(tvm.lower(sch, [A, B, C, E], simple_mode = True))

    D_local = sch.cache_write(D, "local")
    sch[D].set_scope("shared")
    attach_tensor, attach_axis = schedule_bmm(sch, D, C, E, True, True)

    A_shared = sch.cache_read(A, "shared", [D_local])
    A_local = sch.cache_read(A_shared, "local", [D_local])
    B_shared = sch.cache_read(B, "shared", [D_local])
    B_local = sch.cache_read(B_shared, "local", [D_local])

    sch[D_local].compute_at(sch[attach_tensor], attach_axis)
    b, m, l = sch[D_local].op.axis 
    rk, = sch[D_local].op.reduce_axis 
    mm, mi = sch[D_local].split(m, nparts = TY)
    lo, li = sch[D_local].split(l, nparts = TX)
    rko, rki = sch[D_local].split(rk, nparts = TK)
    sch[D_local].reorder(b, mm, lo, rko, rki, mi, li)
    bind(sch[D_local], [(mm, "threadIdx.y"), (lo, "threadIdx.x")])

    sch[A_local].compute_at(sch[D_local], rki)
    sch[B_local].compute_at(sch[D_local], rki)
    sch[A_shared].compute_at(sch[D_local], rko)
    b, m, k = sch[A_shared].op.axis 
    fused = sch[A_shared].fuse(b,m,k)
    fused, ty, tx, vec = tile_axes(sch[A_shared], fused, (TY, TX, VEC))
    bind(sch[A_shared], [(ty, "threadIdx.y"), (tx, "threadIdx.x")])
    sch[A_shared].vectorize(vec)

    sch[B_shared].compute_at(sch[D_local], rko)
    b, k, l = sch[B_shared].op.axis 
    fused = sch[B_shared].fuse(b,k,l)
    fused, ty, tx, vec = tile_axes(sch[B_shared], fused, (TY, TX, VEC))
    bind(sch[B_shared], [(ty, "threadIdx.y"), (tx, "threadIdx.x")])
    sch[B_shared].vectorize(vec)

    # print(tvm.lower(sch, [A, B, C, E], simple_mode = True))
    func = tvm.build(sch, [A, B, C, E], "cuda")
    return [A,B,C], [E], func

time_cost = 0

def test_cuda(batch, M, N, K, L):
    in_dtype = "float16"
    acc_dtype = "float32"
    ins, outs, func = schedule_cuda(
        batch, M, N, K, L, in_dtype=in_dtype, acc_dtype=acc_dtype
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in outs
    ]
    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=300)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    global time_cost 
    time_cost = cost 
    return cost

def getScore(shape):
    x = threading.Thread(target = test_cuda, args = shape)
    global time_cost
    time_cost = math.inf 
    x.start()
    x.join()
    return time_cost

def SA(shape):
    print ("test shape ", shape)
    global TY, TX, TL, TK, WARP_Y, WARP_X, VEC 
    s = [8,8,8,8,8,4,1]
    TY, TX, TL, TK, WARP_Y, WARP_X, VEC = s 
    score = getScore(shape)
    best_score = math.inf
    for k in range(100):
        print ("%d/100: %f ms"%(k, best_score))
        T = 1 - (k+1) / 101
        idx = random.randint(0, len(s)-1)        
        up = s[idx] == 1 or random.random() >= 0.5
        snew = s.copy()
        snew[idx] = snew[idx] * 2 if up else snew[idx] // 2
        TY, TX, TL, TK, WARP_Y, WARP_X, VEC = snew
        score_new = getScore(shape)
        print ("new_score: ", score_new)
        if (math.exp(-(score_new - score) / T) >= random.random()):
            s = snew.copy()
            score = score_new
        best_score = min(best_score, score)
    return best_score

def ceil(x, y):
    return (x + y - 1) // y

def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),  # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)),  # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)),  # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512),  # Mixer-Large/32-S
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "perf"])
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    args = parser.parse_args()
    if args.mode == "correctness":
        test_llvm()
    elif args.mode == "perf":
        costs = []
        for ss in shapes[args.begin : args.begin + args.num]:
            cost = test_cuda(*ss)
            costs.append((ss, cost))

        print("B,M,N,K,L")
        for cc in costs:
            print(
                f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{cc[1]}"
            )
        for cc in costs:
            print(cc[1])
        with open("bmm_bmm_nomk_fuse.pkl", "wb") as f:
            pkl.dump(costs, f)
    else:
        raise ValueError()
