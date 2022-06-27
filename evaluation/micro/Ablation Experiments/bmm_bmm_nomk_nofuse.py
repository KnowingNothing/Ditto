import torch
import tvm 
import numpy as np 
import math 
import argparse
import pickle as pkl

TY_SIZE = 8
TX_SIZE = 8
TK = 8
VEC = 1
WARP_Y = 32
WARP_X = 1
UNROLL_STEP = 512
UNROLL_EXPLICIT = 1
M_factors = [-1, TY_SIZE, WARP_Y]
N_factors = [-1, TX_SIZE, WARP_X]
K_factors = [-1, TK]

def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))

def BatchGemmGemm(batch, M, N, K, L, in_dtype = "float16", acc_dtype = "float32"):
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
    return [A, B, C], [E], D

def schedule_bmm(sch, C):
    A, B = C.op.input_tensors
    
    A_shared = sch.cache_read(A, "shared", [C])
    B_shared = sch.cache_read(B, "shared", [C])
    A_local = sch.cache_read(A_shared, "local", [C])
    B_local = sch.cache_read(B_shared, "local", [C])
    C_local = sch.cache_write(C, "local")

    
    b, m, n = C.op.axis 
    m1, m2, m3 = tile_axes(sch, C, m, M_factors)
    n1, n2, n3 = tile_axes(sch, C, n, N_factors)
    sch[C].reorder(b, m1, n1, m2, n2, m3, n3)
    sch[C].pragma(b, "auto_unroll_max_step", UNROLL_STEP)
    sch[C].pragma(b, "unroll_explicit", UNROLL_EXPLICIT)
    sch[C].bind(b, tvm.te.thread_axis("blockIdx.z"))
    sch[C].bind(m1, tvm.te.thread_axis("blockIdx.y"))
    sch[C].bind(n1, tvm.te.thread_axis("blockIdx.x"))
    sch[C].bind(m2, tvm.te.thread_axis("threadIdx.y"))
    sch[C].bind(n2, tvm.te.thread_axis("threadIdx.x"))

    sch[C_local].compute_at(sch[C], n2)
    b, m, n = C_local.op.axis
    (rk,) = C_local.op.reduce_axis
    rko, rki = tile_axes(sch, C_local, rk, K_factors)
    sch[C_local].reorder(b, rko, rki, m, n)

    def optimize_cache(shared, local):
        sch[shared].compute_at(sch[C_local], rko)
        sch[local].compute_at(sch[C_local], rki)
        b, y, x = sch[shared].op.axis 
        yo, yi = sch[shared].split(y, nparts = TY_SIZE)
        xo, xi = sch[shared].split(x, nparts = TX_SIZE)
        sch[shared].reorder(b, yo, xo, yi, xi)
        sch[shared].bind(yo, tvm.te.thread_axis("threadIdx.y"))
        sch[shared].bind(xo, tvm.te.thread_axis("threadIdx.x"))
    optimize_cache(A_shared, A_local)
    optimize_cache(B_shared, B_local)


def schedule_cuda(batch, M, N, K, L, in_dtype = "float16", acc_dtype = "float32"):
    (A, B, C), (E,), D = BatchGemmGemm(batch, M, N, K, L, in_dtype = "float16", acc_dtype = "float32")
    sch = tvm.te.create_schedule(E.op)
    schedule_bmm(sch, E)
    schedule_bmm(sch, D)
    
    # print(tvm.lower(sch, [A, B, C, E], simple_mode= True))
    func = tvm.build(sch, [A, B, C, E], "cuda")
    return [A, B, C], [E], func

def test_cuda(profile):
    in_dtype = "float16"
    acc_dtype = "float32"
    ins, outs, func = schedule_cuda(
        12, 512, 64, 64, 512, in_dtype=in_dtype, acc_dtype=acc_dtype
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
    if profile:
        func(*inputs_tvm, *outputs_tvm)
    else:
        evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
        cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        print(f"Our code uses {cost} ms")


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

    

