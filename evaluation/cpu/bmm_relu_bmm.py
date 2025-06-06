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

MI = 6
NI1 = 16
KI1 = 64
NI2 = 16
KI2 = 16
REPEAT = 500
PARALLEL = 32
PEAKGFLOPS = int()
WORKLOAD = int()
SERVER = None

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
    'scccc':{
        'parallelism': 36,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,18.0],
        'bandwidth': [465.72, 217.72, 26.54, 14.14],
        'isa': 'avx512',
        'peakgflops': 2995.20
    },
}

def BatchGemmReluGemm(
    batch=1, M=516, N=64, K=64, L=512, dtype="float32"
):
    assert M % MI == 0
    assert N % NI2 == 0
    assert K % KI1 == 0
    assert L % NI1 == 0
    assert L % KI2 == 0

    A = tvm.te.placeholder([batch, M, K], name="A", dtype=dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=dtype)

    A_shared = tvm.te.compute(
        [batch, M // MI, K // KI1, MI, KI1],
        lambda b, mo, ko, mi, ki: A[b, mo * MI + mi, ko * KI1 + ki],
        name="A_shared",
    )

    B_shared = tvm.te.compute(
        [batch, K // KI1, L // NI1, KI1, NI1],
        lambda b, ko, lo, ki, li: B[b, ko * KI1 + ki, lo * NI1 + li],
        name="B_shared",
    )

    rko = tvm.te.reduce_axis([0, K // KI1], "rko")
    rki = tvm.te.reduce_axis([0, KI1], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI1, MI, NI1],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A_shared[b, mo, rko, mi, rki].astype(dtype)
            * B_shared[b, rko, lo, rki, li].astype(dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    relu = tvm.te.compute(
        [batch, M // MI, L // NI1, MI, NI1],
        lambda b, mo, lo, mi, li: tvm.tir.if_then_else(
            D_frag[b, mo, lo, mi, li] > 0, D_frag[b, mo, lo, mi, li], 0
        ).astype(dtype),
        name="relu",
    )

    C_ext = tvm.te.compute(
        [batch, L // KI2, N // NI2, KI2, NI2],
        lambda b, lo, no, li, ni: C[b, lo * KI2 + li, no * NI2 + ni],
        name="C_ext",
    )

    rlo = tvm.te.reduce_axis([0, L // KI2], "rlo")
    rli = tvm.te.reduce_axis([0, KI2], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, N // NI2, MI, NI2],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            relu[b, mo, rlo, mi, rli].astype(dtype)
            * C_ext[b, rlo, no, rli, ni].astype(dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, n // NI2, m %
                               MI, n % NI2].astype(dtype),
        name="F",
    )

    return (
        [A, B, C],
        [F]
    )


doubleGEMMShapes = [
    # (batch, M, N, K, L)
    [8, 516, 512 // 8, 512 // 8, 512],  # Bert-Small
    [12, 516, 768 // 12, 768 // 12, 512],  # Bert-Base
    [16, 516, 1024 // 16, 1024 // 16, 512],  # Bert-Large
    [12, 258, 768 // 12, 768 // 12, 256],  # ViT-Base/14
    [16, 258, 1024 // 16, 1024 // 16, 256],  # ViT-Large/14
    [16, 258, 1280 // 16, 1280 // 16, 256],  # ViT-Huge/14
]


def test_double_gemm(
    batch, M, N, K, L, config={}
):
    print(f"doubleGEMM({batch},{M},{N},{K},{L})")

    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 1.0, 1.0, 0.5, 0.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1
    verbose = False
    isa = "avx2"
    dtype = "float32"
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
    if "verbose" in config:
        verbose = config['verbose']
    if "dtype" in config:
        dtype = config['dtype']
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

    ins, outs = BatchGemmReluGemm(batch, M, N, K, L, dtype=dtype)

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix):
        # cpu_intrin(op, shape, instructionSet, dtype, prefix="")
        packed, code = at.cpu_intrin(
            op="gemm", shape=shape, isa=isa, dtype=dtype, prefix=prefix)
        choices = at.intrinsic_match(op.output(0), packed, [
                                     'InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info

    first_match_info = get_match_info(op1, (MI, NI1, KI1), "gemm1")

    second_match_info = get_match_info(op2, (MI, NI2, KI2), "gemm2")

    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)

    sfs.register_tensorize_axes(tensorizeAxes)

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

    sch0 = tvm.te.create_schedule(outs[0].op)
    func = tvm.build(sch0, layer.schedule_tensors, name="bmm")
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
    
    data = []
    for iter in range(fusionContext.size):
        print("run", iter)
        sch = tvm.te.create_schedule(outs[0].op)

        sch = fusionContext.run(iter, sch, verbose=verbose)

        print(tvm.lower(sch, layer.schedule_tensors, simple_mode = True))

        func = tvm.build(sch, layer.schedule_tensors, name="bmm")

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)

        # for (a, b) in zip(outputs, outputs_tvm):
        #     tvm.testing.assert_allclose(a.numpy(), b.numpy(), atol=1, rtol=1e-6)

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=10, repeat=REPEAT
        )

        cost = evaluator(*inputs_tvm, *outputs_tvm).mean

        predCostList = [float(_) for _ in fusionContext.getPredCostList(iter)]
        
        features = fusionContext.getFeature(iter)
        
        fusionLevel = fusionContext.getFusionLevel(iter)

        print("features", features)

        predCost = fusionContext.getPredCost(iter)

        print("predCostList", predCostList)
        print("cost", cost)
        data.append({'predCostList': predCostList, 'predCost': predCost, 'features': features, 'fusionLevel': fusionLevel, 'cost': cost})
    
    return data


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
        NI1 = 32
        KI1 = 16
        KI2 = 32
        NI2 = 32


def cal_rank_acc(pred: list, label: list):
    print("pred: ", pred)
    print("label: ", label)
    assert len(pred) == len(label)
    acc = 0
    total = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            total += 1
            if abs(label[i] - label[j]) < 1e-5:
                acc += int((pred[i] - pred[j]) < 1e-5)
            else:
                acc += int((pred[i] - pred[j]) * (label[i] - label[j]) > 0)
    return float(acc) / float(total)
all_data = []

def main(batch, M, N, K, L, dtype, server, mode):
    global SERVER
    SERVER = ServerConfig[server]
    setGlobals(batch, M, N, K, L,dtype= dtype)
    print ("B,M,N,K,L,dtype,WORKLOAD,SERVER")
    print(B,M,N,K,L,dtype,WORKLOAD,SERVER)
    global all_data
    all_data.append([])
    data = test_double_gemm(B, M, N, K, L, config={
                           'searchType': 'normal', 'verbose':False, 'mode': mode, 'dtype': dtype})
    with open(f"validation_{batch}_{M}_{N}_{K}_{L}.pkl", 'wb') as f:
        pkl.dump(data, f)
    all_data.append(((batch, M, N, K, L), data))
    predCost = [_['predCost'] for _ in data]
    cost = [_['cost'] for _ in data]
    rank_acc = cal_rank_acc(predCost, cost)
    to_peak = WORKLOAD / 1e9 / cost[0] / SERVER['peakgflops']
    print ("B,M,N,K,L,dtype,WORKLOAD,SERVER,rank_acc,to_peak")
    print(B,M,N,K,L,dtype,WORKLOAD,SERVER,rank_acc,to_peak)
    return {'rank_acc': rank_acc, 'to_peak': to_peak}


example_text = "python ./bmm_bmm_cpu.py --server sc --isa avx2"

shapes = [
    # (batch, M, N, K, L)
    (8, 516, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 516, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 516, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 258, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 258, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 258, 96, 1280 // 16, 256),  # ViT-Huge/14
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
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
        "--server", type=str, choices=['sc', 'sccc', 'scccc']
    )
    parser.add_argument(
        "--mode", type=str, choices=['test', 'survey', 'best']
    )

    args = parser.parse_args()

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
            mode = args.mode
        )
        costs.append((ss, cost))
    with open ("all_data.pkl", 'wb') as f:
        pkl.dump(all_data, f)
    print("B,M,N,K,L,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.dtype},{args.server},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])
