import tvm
import tvm.testing
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
import math
import numpy as np
from ditto.hardware.hw_param import hardware_param
import time
import random
import pickle as pkl
import subprocess
import torch
import regex as re
import math
import numpy as np
import argparse
import os

build_options = [
    "-libs=cblas"
    # '-L/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64',
    # '-L/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/',
    # '-L$HOME/software/llvm/llvm-15.0.0/lib',
    # '-Wl,rpath=$HOME/software/llvm/llvm-15.0.0/lib',
    # '-lmkl_intel_lp64',
    # '-lmkl_intel_thread',
    # '-lmkl_core',
    # '-fopenmp',
    # '-lm',
    # '-lstdc++'
]

MI = 6
NI1 = 16
KI1 = 64
NI2 = 16
KI2 = 16
REPEAT = 200
PEAKGFLOPS = int()
WORKLOAD = int()
SERVER = None

verbose = False

ServerConfig = {
    'sc':{
        'parallelism': 20,
        'cacheSizes': [32 * 16, 32 * 1024, 256 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,10.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx2',
        'peakgflops': 704.20
    },
    'sccc':{
        'parallelism': 32,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,16.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 2150.50
    },
    'scccc': {
        'parallelism': 72,
        'cacheSizes': [32 * 16, 32 * 1024 * 0.8, 1024 * 1024 * 0.8, 25344 * 1024 * 0.8],
        'corePerLevel': [1.0, 1.0, 1.0, 18.0],
        'bandwidth': [293.72, 100.72, 50.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 2995.20
    },
    'Xeon-Gold-6348': {
        'parallelism': 112,
        'cacheSizes': [32 * 32, 2.6 * 1024 * 1024, 70 * 1024 * 1024, 84 * 1024 * 1024],
        'corePerLevel': [1.0, 1.0, 1.0, 28.0],
        'bandwidth': [293.72, 100.72, 50.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 4659.2
    }
}

def BatchGemmGemm(
    batch=1, M=516, N=64, K=64, L=512, dtype="float32"
):
    assert M % MI == 0
    assert N % NI2 == 0
    assert K % KI1 == 0
    assert L % NI1 == 0
    assert L % KI2 == 0

    tensorize_map = {}

    # A = tvm.te.placeholder([batch, M, K], name="A", dtype=dtype)
    A = tvm.te.placeholder([batch, M // MI, MI, K // KI1, KI1], name="A", dtype=dtype)
    #  B = tvm.te.placeholder([batch, M, L], name="B", dtype=dtype)
    B = tvm.te.placeholder([batch, K // KI1, KI1, L // NI1, NI1], name="B", dtype=dtype)
    #  C = tvm.te.placeholder([batch, L, N], name="C", dtype=dtype)
    C = tvm.te.placeholder([batch, L // KI2, KI2, N // NI2, NI2], name="C", dtype=dtype)

    rko = tvm.te.reduce_axis([0, K // KI1], "rko")
    rki = tvm.te.reduce_axis([0, KI1], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, MI, L // NI1, NI1],
        lambda b, mo, mi, lo, li: tvm.te.sum(
            A[b, mo, mi, rko, rki].astype(dtype)
            * B[b, rko, rki, lo, li].astype(dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    tensorize_map.update({D_frag.op.axis[1]: D_frag.op.axis[2], D_frag.op.axis[3]:D_frag.op.axis[4], rko: rki})
    choice1 = [D_frag.op.axis[2], D_frag.op.axis[4], rki]

    # C_ext = tvm.te.compute(
    #     [batch, L // KI2, N // NI2, KI2, NI2],
    #     lambda b, lo, no, li, ni: C[b, lo * KI2 + li, no * NI2 + ni],
    #     name="C_ext",
    # )

    rlo = tvm.te.reduce_axis([0, L // KI2], "rlo")
    rli = tvm.te.reduce_axis([0, KI2], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, MI, N // NI2, NI2],
        lambda b, mo, mi, no, ni: tvm.te.sum(
            D_frag[b, mo, mi, rlo, rli].astype(dtype)
            * C[b, rlo, rli, no, ni].astype(dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    tensorize_map.update({E_frag.op.axis[1]: E_frag.op.axis[2], E_frag.op.axis[3]: E_frag.op.axis[4], rlo: rli})
    choice2 = [E_frag.op.axis[2], E_frag.op.axis[4], rli]

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, m %
                               MI, n // NI2,  n % NI2].astype(dtype),
        name="F",
    )

    return (
        [A, B, C],
        [F],
        (choice1, choice2),
        tensorize_map
    )

def test_double_gemm(
    batch, M, N, K, L, config={}
):
    print(f"doubleGEMM({batch},{M},{N},{K},{L})")

    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0, 1.0, 1.0, 1.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1
    isa = "avx2"
    dtype = "float32"
    topk = 5
    mk_type = ""
    if SERVER:
        cacheSizes = SERVER['cacheSizes']
        bandwidth = SERVER['bandwidth']
        parallelism = SERVER['parallelism']
        isa = SERVER['isa']
    if "tensorWeight" in config:
        tensorWeight = config["tensorWeight"]
    if "searchType" in config:
        searchType = config["searchType"]
    if "mode" in config:
        mode = config["mode"]
    if "dtype" in config:
        dtype = config['dtype']
    if 'topk' in config:
        topk = config['topk']
    if 'mk_type' in config:
        mk_type = config['mk_type']
    print("begin test ...")
    print("shape,cacheSizes,bandwidth,tensorWeight,searchType,mode,parallelism,isa,dtype")
    print(batch, M, N, K, L, cacheSizes, bandwidth, tensorWeight,
          searchType, mode, parallelism, isa, dtype)
    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=-1,
        num_groups=parallelism,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=cacheSizes,
        bandwidth=bandwidth,
        coresPerCacheLevel=[1.0, 1.0, 1.0, 16.0],
        tensorWeight=tensorWeight,
        platform="CPU",
    )

    ins, outs, (choice1, choice2), tensorize_map = BatchGemmGemm(batch, M, N, K, L, dtype=dtype)

    t0 = time.time()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix, choice = None):
        # cpu_intrin(op, shape, instructionSet, dtype, prefix="")
        packed, code = at.cpu_intrin(
            op="gemm_noreshape", shape=shape, isa=isa, dtype=dtype, prefix=prefix, surfix = mk_type)
        if choice == None:
            choices = at.intrinsic_match(op.output(0), packed, [
                                        'SameRange'])
            choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info
    

    first_match_info = get_match_info(op1, (MI, NI1, KI1), "gemm1", choice=choice1)

    second_match_info = get_match_info(op2, (MI, NI2, KI2), "gemm2", choice = choice2)

    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)

    sfs.register_tensorize_axes(tensorizeAxes)
    sfs.register_tensorize_map(tensorize_map)

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype=dtype, simple_mode=-1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

    data_path = ""
    res_path = ""

    print("mode is ", mode)
    fusionContext = at.build_fusion_context(
        sfs,
        layer,
        tensorize_state,
        data_path,
        hw_param=CPU,
        dtype=dtype,
        searchType=searchType,
        mode=mode,
    )

    t1 = time.time()
    print("static analysis time", t1-t0)
    sch0 = tvm.te.create_schedule(outs[0].op)
    func = tvm.build(sch0, layer.schedule_tensors, name="bmm", target = "llvm " + ' '.join(build_options))
    ctx = tvm.cpu()
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in ins
    ]
    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in outs
    ]
    inputs = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs = [tvm.nd.array(x, ctx) for x in outputs_np]
    func(*inputs, *outputs)

    top = float()
    topk_time = float('inf')
    if mode == "test":
        R = 1
        stepsize = 1
    elif mode == "best":
        R = min(topk, fusionContext.size)
        stepsize = 1
    elif mode == "survey":
        R = min(topk, fusionContext.size) 
        stepsize = max(1, int(R / topk))

    fusionLevel = fusionContext.getFusionLevel(0)
    times = []
    for iter in range(0, R, stepsize):
        print("run", iter)
        sch = tvm.te.create_schedule(outs[0].op)

        sch = fusionContext.run(iter, sch, verbose=verbose)

        func = tvm.build(sch, layer.schedule_tensors, name="bmm", target = "llvm " + ' '.join(build_options))

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=0, repeat=REPEAT, number = 1, f_preproc="cache_flush_cpu_non_first_arg"
        )


        cost = evaluator(*inputs_tvm, *outputs_tvm).mean
        times.append(cost)
        if iter == 0:
            top = cost
        if cost < topk_time:
            topk_time = cost
        print(f'iter{iter}: ', cost)
    return {'top1': top, f'top{topk}': topk_time, 'geomean': np.exp(np.mean(np.log(times)))}, fusionLevel, t1 - t0, times


def setGlobals(B, M, N, K, L, dtype):
    global MI, NI1, KI1, NI2, KI2, WORKLOAD, PEAKGFLOPS
    WORKLOAD = B * M * L * (K + N)
    if SERVER['isa'] == "avx2":
        MI = 6
        NI1 = 16
        KI1 = math.gcd(K, L)
        KI2 = 16
        NI2 = 16
    elif SERVER['isa'] == "avx512":
        MI = 6
        NI1 = 64 if N % 64 == 0 else 32 
        KI1 = 64 if K % 64 == 0 else K
        KI2 = NI1
        NI2 = 64 if L % 64 == 0 else 32
    M = uround(M, MI)
    N = uround(N, NI2)
    L = uround(L, NI1)
    return (B, M, N, K, L)


def main(batch, M, N, K, L, dtype, server, mode, weights, topk, search_type, mk_type):
    global SERVER
    SERVER = ServerConfig[server]
    B, M, N, K, L = setGlobals(batch, M, N, K, L,dtype= dtype)
    print ("B,M,N,K,L,dtype,WORKLOAD,SERVER")
    print(B,M,N,K,L,dtype,WORKLOAD,SERVER)
    t0 = time.time()
    Time, fusionLevel, schedule_time, all_times = test_double_gemm(B, M, N, K, L, config={
                           'searchType': search_type, 'mode': mode, 
                           'dtype': dtype, 'tensorWeight': weights,
                           'topk': topk, 'mk_type': mk_type})
    t1 = time.time()
    print("schedule time: ", t1 - t0)
    ret = {}
    for k in Time:
        ret[k] = {} 
        ret[k]['time(s)'] = Time[k]
        ret[k]['%peak'] = (WORKLOAD / Time[k] / 1e9) / SERVER['peakgflops']
    ret['std'] = np.std([(WORKLOAD / x / 1e9 / SERVER['peakgflops']) for x in all_times])
    ret['n_sample'] = len(all_times)
    ret['total_time'] = t1 - t0
    ret['fusionLevel'] = fusionLevel

    print(batch, M, N, K, L, dtype,":",ret)
    return ret


example_text = "python ./bmm_bmm_cpu.py --server sc "

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mk_type", 
        type = str, 
        choices = ['mkl', 'baseline', ''],
        default = "",
        help = "whether a naive microkernel is used"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", 'float64'],
        default="float32",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--server", type=str, choices=ServerConfig.keys()
    )
    parser.add_argument(
        "--mode", type=str, choices=['test', 'best', 'survey'], default = "test"
    )

    parser.add_argument(
        "--store", type=bool, choices = [0, 1], default = 1
    )
    
    parser.add_argument(
        "--weights", type=float, nargs="+", default=[1.0,1.0,1.0,1.0,1.0]
    )

    parser.add_argument(
        "--topk", type=int, default=5
    )

    parser.add_argument(
        "--search_type", type=str, choices = ['normal', 'stochastic', 'nofuse'], default = "normal"
    )

    parser.add_argument(
        "--output", type = str, default = "result"
    )

    args = parser.parse_args()
    
    os.system(f"mkdir -p {args.output}")

    verbose = args.verbose

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(
            batch=B,
            M=M,
            N=N,
            K=K,
            L=L,
            dtype=args.dtype,
            server = args.server,
            mode = args.mode,
            weights = args.weights,
            topk = args.topk,
            search_type = args.search_type,
            mk_type = args.mk_type
        )
        costs.append((ss, cost))
        if args.store:
            with open(f"{args.output}/bmm_bmm-chimera.pkl", 'wb') as f:
                pkl.dump(costs, f)

    print("B,M,N,K,L,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.dtype},{args.server},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])
    

'''
No-C(easy): 
    fusion: loop structure; 
    tiling is defined by chimera/lirui's model
No-F(easy):   
    fusion at outermost cache level;
No-M(easy):
'''
