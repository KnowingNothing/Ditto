import tvm 
from ditto import auto_tensorize as at 
from ditto.hardware.hw_param import hardware_param
import numpy as np
import argparse 
import pickle as pkl 



MI = 6
NI1 = 32
KI1 = 64
NI2 = 32
KI2 = 32
isa = "avx512"
dtype = "float32"

def BatchGemmGemm(
    batch=12, M=516, N=64, K=64, L=512, dtype="float32"
):
    assert M % MI == 0
    assert N % NI2 == 0
    assert K % KI1 == 0
    assert L % NI1 == 0
    assert L % KI2 == 0

    # A = tvm.te.placeholder([batch, M, K], name="A", dtype=dtype)
    A = tvm.te.placeholder([batch, M // MI, MI, K // KI1, KI1], name="A", dtype=dtype)
    #  B = tvm.te.placeholder([batch, M, L], name="B", dtype=dtype)
    B = tvm.te.placeholder([batch, K // KI1, KI1, L // NI1, NI1], name="B", dtype=dtype)
    #  C = tvm.te.placeholder([batch, L, N], name="C", dtype=dtype)
    C = tvm.te.placeholder([batch, L // KI2, KI2, N // NI2, NI2], name="C", dtype=dtype)

    rko = tvm.te.reduce_axis([0, K // KI1], "rko")
    rki = tvm.te.reduce_axis([0, KI1], "rki")
    D = tvm.te.compute(
        [batch, M // MI, MI, L // NI1, NI1],
        lambda b, mo, mi, lo, li: tvm.te.sum(
            A[b, mo, mi, rko, rki].astype(dtype)
            * B[b, rko, rki, lo, li].astype(dtype),
            axis=[rko, rki],
        ),
        name="D",
    )

    choice1 = [D.op.axis[2], D.op.axis[4], rki]

    rlo = tvm.te.reduce_axis([0, L // KI2], "rlo")
    rli = tvm.te.reduce_axis([0, KI2], "rli")
    E = tvm.te.compute(
        [batch, M // MI, MI, N // NI2, NI2],
        lambda b, mo, mi, no, ni: tvm.te.sum(
            D[b, mo, mi, rlo, rli].astype(dtype)
            * C[b, rlo, rli, no, ni].astype(dtype),
            axis=[rlo, rli],
        ),
        name="E",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E[b, m // MI, m % MI, n // NI2, n % NI2],
        'F'
    )

    choice2 = [E.op.axis[2], E.op.axis[4], rli]


    return [A, B, C], [D, E, F],(choice1, choice2)

def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))

def schedule_single_op(sch, tensor, tensorize_axes, shape, prefix):
    packed, code = at.cpu_intrin(
                op="gemm_noreshape", shape=shape, isa=isa, dtype=dtype, prefix=prefix)
    b, m, mt, n, nt = sch[tensor].op.axis 
    rko, rkt = sch[tensor].op.reduce_axis 
    mo, mm, mi = tile_axes(sch, tensor, m, [-1, 2, 4])
    no, nm, ni = tile_axes(sch, tensor, n, [-1, 2, 4])
    rk1, rk2 = tile_axes(sch, tensor, rko, [-1,4])
    sch[tensor].reorder(b, mo, no, mm, nm, rk1, mi, ni, rk2, mt, nt, rkt)
    fused = sch[tensor].fuse(b, mo, no)
    sch[tensor].parallel(fused)
    sch[tensor].tensorize(mt, packed.compute_intrinsic)
    sch[tensor].pragma(fused, "import_llvm", code)
    return sch 
def main(shape, export_lib = False):
    ins, outs, choices = BatchGemmGemm(*shape)
    A, B, C = ins 
    D, E, F = outs 
    choice1, choice2 = choices
    sch = tvm.te.create_schedule(F.op)

    axes = F.op.axis 
    fused = sch[F].fuse(*axes)
    parallel, fused = sch[F].split(fused, nparts = 72)
    fused, vec = sch[F].split(fused, 16)
    sch[F].parallel(parallel)
    sch[F].vectorize(vec)

    schedule_single_op(sch, E.op, choice2, (MI, NI2, KI2), "E")
    print("success schedule second op")
    schedule_single_op(sch, D.op, choice1, (MI, NI1, KI1), "D")
    print("success schedule first op")
    print(tvm.lower(sch, [A, B, C, F], simple_mode = True))
    func = tvm.build(sch, [A, B, C, F], name="bmm")
    ctx = tvm.cpu()
    inputs_tvm = [tvm.nd.array(np.random.uniform(-1,1,[int(_)for _ in x.shape]).astype(x.dtype), ctx) for x in [A, B, C]]
    outputs_tvm = [tvm.nd.array(np.zeros(shape = [int(_) for _ in x.shape]).astype(x.dtype), ctx) for x in [F]]

    func(*inputs_tvm, *outputs_tvm)

    evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=0, repeat=500, number = 1, f_preproc="cache_flush_cpu_non_first_arg"
        )

    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    print("our code uses: %f ms"% cost)
    batch, M, N, K, L = shape
    gflops = batch * M * L * (N + K) / cost / 1e6 
    ret = {
        'cost': cost,
        'gflops': gflops,
        'toPeak': gflops / 2995.2,
    }
    if export_lib:
        libname = f"lib_nofuse_{batch}_{M}_{N}_{K}_{L}.so"
        func.export_library('./lib/' + libname)
        ret['libname'] = libname 
        ret['tensorInfo'] = [([int(_) for _ in x.shape], x.dtype) for x in [A,B,C,F]]
    return ret

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (16, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (24, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512), # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),   # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256), # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256), # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),   # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)), # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)), # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512), # Mixer-Large/32-S
]

def setGlobals(B, M, N, K, L):
    global MI, NI1, KI1, NI2, KI2
    MI = 6
    NI1 = 64 if N % 64 == 0 else 32 
    KI1 = 64 if K % 64 == 0 else K
    KI2 = NI1
    NI2 = 64 if L % 64 == 0 else 32
    M = uround(M, MI)
    N = uround(N, NI2)
    L = uround(L, NI1)
    return (B, M, N, K, L)

example_text = "python ./bmm_bmm_cpu.py --begin 0 --num 1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")

    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    parser.add_argument(
        "--store", action = "store_true"
    )

    parser.add_argument(
        "--export_lib", action = "store_true"
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        ss = setGlobals(*ss)
        cost = main(ss, export_lib = args.export_lib)
        costs.append((ss, cost))

    print("B,M,N,K,L,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])
    
    if args.store:
        with open("bmm_bmm_nofuse_cpu.pkl", 'wb') as f:
            pkl.dump(costs, f)