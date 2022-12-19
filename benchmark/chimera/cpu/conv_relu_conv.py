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
import regex as re
import math
import numpy as np
import argparse
import pickle as pkl
import os

class MicroKernel:
    def __init__(self, N=1, K=4, H=3, W=8, C=1, R=3, S=3) -> None:
        self.N = N
        self.K = K
        self.H = H
        self.W = W
        self.C = C
        self.R = R
        self.S = S

    def RoundUp(self, N, K, H, W, C, R, S):
        def f(a, b):
            return (a + b - 1) // b * b

        return [
            f(N, self.N),
            f(K, self.K),
            f(H, self.H),
            f(W, self.W),
            f(C, self.C),
            f(R, self.R),
            f(S, self.S),
        ]

    def Verify(self, N, K, H, W, C, R, S):
        return (
            N % self.N == 0
            and K % self.K == 0
            and H % self.H == 0
            and W % self.W == 0
            and R % self.R == 0
            and S % self.S == 0
        )


mk1 = None
mk2 = None
REPEAT = 2000
PARALLEL = 32
WORKLOAD = int()
SERVER = None

ServerConfig = {
    'sc': {
        'parallelism': 20,
        'cacheSizes': [32 * 16, 32 * 1024, 256 * 1024, 25344 * 1024],
        'corePerLevel': [1.0, 1.0, 1.0, 10.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx2',
        'peakgflops': 704.20
    },
    'sccc': {
        'parallelism': 32,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024, 25344 * 1024],
        'corePerLevel': [1.0, 1.0, 1.0, 16.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 2150.50
    },
    'scccc': {
        'parallelism': 36,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024 * 0.8, 25344 * 1024 * 0.8],
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


# 1,128,388,388,64,3,3,64,3,3

def ConvReluConv(
    shape,
    dtype="float32"
):
    """
            N1 K4 H3 W16 C1 R3 S3
    Conv1   N C1 P1 Q1 C0 R1 S1
    Conv1   N C2 P2 Q2 C1 R2 S2
    """
    assert len(shape) == 14
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    P0_pad = P0 + 2 * padding1
    Q0_pad = Q0 + 2 * padding1
    P1 = (P0_pad - R1) // stride1 + 1
    Q1 = (Q0_pad - S1) // stride1 + 1
    P1_pad = P1 + 2 * padding2
    Q1_pad = Q1 + 2 * padding2
    P2 = (P1_pad - R2) // stride2 + 1
    Q2 = (Q1_pad - S2) // stride2 + 1
    global mk1, mk2
    assert C1 % mk1.K == 0
    assert C2 % mk2.K == 0
    assert P1 % mk1.H == 0
    assert P2 % mk2.H == 0
    assert Q1 % mk1.W == 0
    assert Q2 % mk2.W == 0
    assert C0 % mk1.C == 0
    assert C1 % mk2.C == 0
    
    Img = tvm.te.placeholder([N, C0, P0, Q0], dtype=dtype, name="Img")
    Weight1 = tvm.te.placeholder([C1, C0, R1, S1], dtype=dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([C2, C1, R2, S2], dtype=dtype, name="Weight2")

    Pad1 = tvm.te.compute(
        [N, C0, P0_pad, Q0_pad],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding1, p < P0 + padding1, q >= padding1, q < Q0 + padding1
            ),
            Img[n, c, p - padding1, q - padding1],
            tvm.tir.const(0, dtype),
        ),
        name="pad1",
    )
    r1 = tvm.te.reduce_axis((0, R1), name="rr1")
    s1 = tvm.te.reduce_axis((0, S1), name="rs1")
    co1 = tvm.te.reduce_axis((0, C0 // mk1.C), name="rco1")
    

    ci1 = tvm.te.reduce_axis((0, mk1.C), name="rci1")

    Conv1 = tvm.te.compute(
        [N // mk1.N, mk1.N, C1 // mk1.K, mk1.K, P1 // mk1.H, mk1.H, Q1 // mk1.W, mk1.W],
        lambda no1, ni1, ko1, ki1, ho1, hi1, wo1, wi1: tvm.te.sum(
            Pad1[
                no1 * mk1.N + ni1,
                co1 * mk1.C + ci1,
                ho1 * mk1.H + hi1 + r1,
                wo1 * mk1.W + wi1 + s1,
            ]
            * Weight1[ko1 * mk1.K + ki1, co1 * mk1.C + ci1, r1, s1],
            axis=[co1, ci1, r1, s1],
        ),
        name="conv1",
    )
    choice1 = [Conv1.op.axis[_]for _ in [1,3,5,7]] + [ci1, r1, s1]
    Conv1_rfact = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: Conv1[
            n // mk1.N,
            n % mk1.N,
            c // mk1.K,
            c % mk1.K,
            p // mk1.H,
            p % mk1.H,
            q // mk1.W,
            q % mk1.W,
        ],
        name="conv1_unfactored",
    )
    Conv1_relu = tvm.te.compute(
        [N, C1, P1, Q1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            Conv1_rfact[n, c, p, q] > 0,
            Conv1_rfact[n, c, p, q],
            tvm.tir.const(0, dtype),
        ),
        name="relu",
    )

    Pad2 = tvm.te.compute(
        [N, C1, P1_pad, Q1_pad],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding2, p < P0 + padding2, q >= padding2, q < Q0 + padding2
            ),
            Conv1_relu[n, c, p - padding2, q - padding2],
            tvm.tir.const(0, dtype),
        ),
        name="pad2",
    )
    r2 = tvm.te.reduce_axis((0, R2), name="rr2")
    s2 = tvm.te.reduce_axis((0, S2), name="rs2")
    co2 = tvm.te.reduce_axis((0, C1 // mk2.C), name="rco2")
    ci2 = tvm.te.reduce_axis((0, mk2.C), name="rci2")
    Conv2_fact = tvm.te.compute(
        [N // mk2.N, mk2.N, C2 // mk2.K, mk2.K, P2 // mk2.H, mk2.H, Q2 // mk2.W, mk2.W],
        lambda no2, ni2, ko2, ki2, ho2, hi2, wo2, wi2: tvm.te.sum(
            Pad2[
                no2 * mk2.N + ni2,
                co2 * mk2.C + ci2,
                ho2 * mk2.H + hi2 + r2,
                wo2 * mk2.W + wi2 + s2,
            ]* Weight2[ko2 * mk2.K + ki2, co2 * mk2.C + ci2, r2, s2],
            axis=[co2, ci2, r2, s2],
        ).astype(dtype),
        name="conv2_fact",
    )
    choice2 = [Conv2_fact.op.axis[_]for _ in [1,3,5,7]] + [ci2, r2, s2]
    Conv2 = tvm.te.compute(
        [N, C2, P2, Q2],
        lambda n, c, p, q: Conv2_fact[
            n // mk2.N,
            n % mk2.N,
            c // mk2.C,
            c % mk2.C,
            p // mk2.H,
            p % mk2.H,
            q // mk2.W,
            q % mk2.W,
        ],
        name="conv2",
    )
    return (
        [Img, Weight1, Weight2],
        [Conv2],
        (choice1, choice2, choice1+choice2)
    )

def test_double_conv(
    shape, config={}
):
    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1
    verbose = False
    isa = "avx2"
    dtype = "float32"
    topn = 10
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
    if "topn" in config:
        topn = config['topn']
    print("begin test ...")
    print("shape,cacheSizes,bandwidth,tensorWeight,searchType,mode,parallelism,isa,dtype")
    print(shape, cacheSizes, bandwidth, tensorWeight,
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

    ins, outs, (choice1, choice2, tensorizeAxes) = ConvReluConv(
        shape, dtype=dtype)

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op
    # def get_match_info(op, shape, prefix):
    #     # cpu_intrin(op, shape, instructionSet, dtype, prefix="")
    #     packed, code = at.cpu_intrin(
    #         op="conv", shape=shape, isa=isa, dtype=dtype, prefix=prefix)
    #     choices = at.intrinsic_match(op.output(0), packed, [])
    #     choice = choices[0]
    #     match_info = at.match_info(choice, packed, code)
    #     return match_info

    # first_match_info = get_match_info(op1, (mk1.N, mk1.K, mk1.H, mk1.W, mk1.C, mk1.R,mk1.S), "conv1")

    # second_match_info = get_match_info(op2, (mk2.N, mk2.K, mk2.H, mk2.W, mk2.C, mk2.R, mk2.S), "conv2")

    # tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)

    packed1, code1 = at.cpu_intrin(op="conv", shape=(
        mk1.N, mk1.K, mk1.H, mk1.W, mk1.C, mk1.R, mk1.S), isa=isa, dtype=dtype, prefix="conv1")
    first_match_info = at.match_info(choice1, packed1, code1)

    packed2, code2 = at.cpu_intrin(op="conv", shape=(
        mk2.N, mk2.K, mk2.H, mk2.W, mk2.C, mk2.R, mk2.S), isa=isa, dtype=dtype, prefix="conv2")
    second_match_info = at.match_info(choice2, packed2, code2)

    sfs.register_tensorize_axes(tensorizeAxes)

    print("begin building fusion choice...")
    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype=dtype, simple_mode=1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

    data_path = ""

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
    func = tvm.build(sch0, layer.schedule_tensors, name="conv_relu_conv")
    ctx = tvm.cpu()
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in ins
    ]
    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in outs
    ]
    inputs = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs = [tvm.nd.array(x, ctx) for x in outputs_np]
    func(*inputs, *outputs)

    top = float()
    topn_time = float('inf')
    if mode == "best":
        R = fusionContext.size if fusionContext.size < topn else topn
    elif mode =="test":
        R = 1
    for iter in range(R):
        print("run", iter)
        sch = tvm.te.create_schedule(outs[0].op)

        sch = fusionContext.run(iter, sch, verbose=verbose)

        print ("schedule success!")
        if (mode == "test"):
            lowerCode = tvm.lower(sch, layer.schedule_tensors, simple_mode = True)
            with open ("tmp.txt", 'w') as f:
                f.write(str(lowerCode))
        
        func = tvm.build(sch, layer.schedule_tensors, name="conv_relu_conv")

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)

        # for (a, b) in zip(outputs, outputs_tvm):
        #     tvm.testing.assert_allclose(
        #         a.numpy(), b.numpy(), atol=1, rtol=1e-6)
        
        # print("test passed!")

        evaluator = func.time_evaluator(
            func.entry_name, ctx, repeat = 1, number = 10000
        )

        print("begin evaluate ...")
        cost = evaluator(*inputs_tvm, *outputs_tvm)

        cost = np.mean(cost.results[100:])
        
        if iter == 0:
            top = cost
        if mode == "test":
            break
        if cost < topn_time:
            topn_time = cost
        print(f'{shape} iter{iter}: ', cost)
    return {'top1': top, 'topn': topn_time}


def setGlobals(shape):
    global MI, NI1, KI1, NI2, KI2, WORKLOAD, mk1, mk2
    N, C0, H, W, C1, R1, S1, C2, R2, S2,padding1,padding2, stride1,stride2 = shape
    assert R1 == 3 and R2 == 3
    WORKLOAD = N * (C0 * H * W * C1 * R1 * S1 + C1 * H * W * C2 * R2 * S2)
    if SERVER['isa'] == "avx2":
        mk1 = MicroKernel(N=1, K=4, H=3, W=8, C=16, R=3, S=3)
        mk2 = MicroKernel(N=1, K=4, H=3, W=8, C=4, R=3, S=3)
    elif SERVER['isa'] == "avx512":
        mk1 = MicroKernel(N=1, K=4, H=3, W=32, C=64, R=3, S=3)
        mk2 = MicroKernel(N=1, K=4, H=3, W=32, C=64, R=3, S=3)


def main(shape, dtype, server):
    global SERVER
    SERVER = ServerConfig[server]
    setGlobals(shape)
    print("shape,dtype,WORKLOAD,SERVER")
    print(shape, dtype, WORKLOAD, SERVER)
    time = test_double_conv(shape, config={
        'searchType': 'normal', 'verbose': True, 'mode': 'best', 'dtype': dtype, 'topn': 5})
    ret = {}
    for k in time:
        ret[k] = {}
        ret[k]['time(s)'] = time[k]
        ret[k]['%peak'] = (WORKLOAD / time[k] / 1e9) / SERVER['peakgflops']
        ret[k]['gflops'] = (WORKLOAD / time[k] / 1e9)
    print(shape, dtype, ":", ret)
    return ret


example_text = "python ./conv_relu_conv.py --server sc --mode best"

shapes = [
    [1, 64, 114, 112, 192, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 32, 147, 148, 64, 3, 3, 96, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 128, 3, 3, 64, 1, 1, 1, 0, 1, 1],
    [1, 128, 27, 28, 256, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 16, 228, 227, 64, 3, 3, 32, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 64, 64, 1, 1, 64, 3, 3, 0, 1, 1, 1],
    [1, 64, 57, 56, 64, 1, 1, 64, 1, 1, 0, 0, 1, 1],
    [1, 256, 57, 56, 256, 1, 1, 64, 1, 1, 0, 0, 1, 1]
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
        "--server", type=str, choices=ServerConfig.keys()
    )
    
    parser.add_argument(
        '--output', type=str, default='result'
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        cost = main(
            ss,
            dtype=args.dtype,
            server=args.server
        )
        costs.append((ss, cost))

    print("shape,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.dtype},{args.server},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])
        
    os.system(f"mkdir -p {args.output}")
    with open(f"{args.output}/conv_relu_conv-chimera.pkl", "wb") as f:
        pkl.dump(costs, f)
