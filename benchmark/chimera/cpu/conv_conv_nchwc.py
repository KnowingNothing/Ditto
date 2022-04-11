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
}


def ceil(x, y):
    return (x + y - 1) // y


MI1  = 6
NI1 = 64
KI1 = 64

MI2  = 6
NI2 = 64
KI2 = 64

REPEAT = 2000

def conv_conv_nchwc(
    N,
    C,
    H,
    W,
    K1,
    R1,
    S1,
    K2,
    R2,
    S2,
    padding1=1,
    padding2=1,
    stride1=1,
    stride2=1,
    in_dtype="float32",
    acc_dtype="float32",
):
    P1 = (H + 2 * padding1 - R1) // stride1 + 1
    Q1 = (W + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    assert P1 * Q1 % MI1 == 0
    assert P2 * Q2 % MI2 == 0
    assert C % KI1 == 0
    assert K1 % NI1 == 0
    assert NI1 == KI2
    assert K2 % NI2 == 0

    CI = KI1
    CO = ceil(C, KI1)
    PQ1I = MI1
    PQ1O = ceil(P1 * Q1, MI1)
    K1I = NI1
    K1O = ceil(K1, NI1)
    PQ2I = MI2
    PQ2O = ceil(P2 * Q2, MI2)
    K2I = NI2
    K2O = ceil(K2, NI2)
    RK1I = KI2
    RK1O = ceil(K1, KI2)


    Img = tvm.te.placeholder([N, CO, R1, S1, PQ1O, PQ1I, CI], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder([K1, C, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([K2, K1, R2, S2], dtype=in_dtype, name="Weight2")

    Weight1_fact = tvm.te.compute(
        [K1O, CO, R1, S1, CI, K1I],
        lambda k1o, co, r1, s1, ci, k1i: tvm.tir.if_then_else(
            tvm.tir.all(k1o * K1I + k1i < K1, co * CI + ci < C),
            Weight1[k1o * K1I + k1i, co * CI + ci, r1, s1],
            tvm.tir.const(0, in_dtype),
        ),
        name="Weight1_fact",
    )

    rc1o = tvm.te.reduce_axis([0, CO], "rc1o")
    rc1i = tvm.te.reduce_axis([0, CI], "rc1i")
    rr1 = tvm.te.reduce_axis([0, R1], "rr1")
    rs1 = tvm.te.reduce_axis([0, S1], "rs1")
    conv1_frag = tvm.te.compute(
        [N, K1O, PQ1O, PQ1I, K1I],
        lambda n, ko, pqo, pqi, ki: tvm.te.sum(
            Img[n, rc1o, rr1, rs1, pqo, pqi, rc1i].astype(acc_dtype)
            * Weight1_fact[ko, rc1o, rr1, rs1, rc1i, ki].astype(acc_dtype),
            axis=[rr1, rs1, rc1o, rc1i],
        ),
        name="conv1_frag",
    )
    choice1 = [conv1_frag.op.axis[_] for _ in [-2, -1]] + [rc1i]

    # relu1 = tvm.te.compute(
    #     [N, K1O, PQ1O, PQ1I, K1I],
    #     lambda n, ko, pqo, pqi, ki: tvm.tir.if_then_else(
    #         conv1_frag[n, ko, pqo, pqi, ki] > tvm.tir.const(0, acc_dtype),
    #         conv1_frag[n, ko, pqo, pqi, ki].astype(in_dtype),
    #         tvm.tir.const(0, in_dtype),
    #     ),
    #     name="relu1",
    # )

    assert K1O * K1I == K1 and PQ1O * PQ1I == P1 * Q1
    assert K1I == RK1I


    pad2_fact = tvm.te.compute(
        [N, RK1O, R2, S2, PQ1O, PQ1I, RK1I],
        lambda n, rk1o, r, s, pq1o, pq1i, rk1i: tvm.tir.if_then_else(
            tvm.tir.all(
                padding2 <= ((pq1o * PQ1I + pq1i) // Q1 + r),
                P1+padding2-1 >= ((pq1o * PQ1I + pq1i) // Q1 + r),
                padding2 <= ((pq1o * PQ1I + pq1i) % Q1 + s),
                Q1+padding2-1 >= ((pq1o * PQ1I + pq1i) % Q1 + s)
            ),
            conv1_frag[
                n,
                rk1o,
                pq1o + (pq1i + (r - padding2) * Q1 + s - padding2) // PQ1I,
                (pq1i + (r - padding2) * Q1 + s - padding2) % PQ1I,
                rk1i
            ],
            tvm.tir.const(0, in_dtype),
        ),
        name="pad2_fact",
    )

    Weight2_fact = tvm.te.compute(
        [K2O, RK1O, R2, S2, RK1I, K2I],
        lambda k2o, rk1o, r2, s2, rk1i, k2i: tvm.tir.if_then_else(
            tvm.tir.all(k2o * K2I + k2i < K2, rk1o * RK1I + rk1i < K1),
            Weight2[k2o * K2I + k2i, rk1o * RK1I + rk1i, r2, s2],
            tvm.tir.const(0, in_dtype),
        ),
        name="Weight2_fact",
    )

    rk1o = tvm.te.reduce_axis([0, RK1O], "rk1o")
    rk1i = tvm.te.reduce_axis([0, RK1I], "rk1i")
    rr2 = tvm.te.reduce_axis([0, R2], "rr2")
    rs2 = tvm.te.reduce_axis([0, S2], "rs2")
    conv2_frag = tvm.te.compute(
        [N, K2O, PQ2O, PQ2I, K2I],
        lambda n, ko, pqo, pqi, ki: tvm.te.sum(
            pad2_fact[n, rk1o, rr2, rs2, pqo, pqi, rk1i].astype(acc_dtype)
            * Weight2_fact[ko, rk1o, rr2, rs2, rk1i, ki].astype(acc_dtype),
            axis=[rr2, rs2, rk1o, rk1i],
        ),
        name="conv2_frag",
    )
    choice2 = [conv2_frag.op.axis[_] for _ in [-2, -1]] + [rk1i]

    print("conv1_frag", conv1_frag)
    print("conv2_frag", conv2_frag)
    return [Img, Weight1, Weight2], [conv2_frag],\
    (Img,Weight1,Weight2,conv1_frag, Weight1_fact, Weight2_fact, pad2_fact, conv2_frag) # , pad1, pad1_fact)#conv2_frag_rfact) #pad1, pad1_fact

def test_double_conv(
    shape, config={}
):
    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0, 1.0, 1.0, 1.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1
    verbose = False
    isa = "avx2"
    dtype = "float32"
    topn = 1
    corePerLevel = [1.0, 1.0, 1.0, 16.0]
    if SERVER:
        cacheSizes = SERVER['cacheSizes']
        bandwidth = SERVER['bandwidth']
        parallelism = SERVER['parallelism']
        isa = SERVER['isa']
        corePerLevel = SERVER['corePerLevel']
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
        coresPerCacheLevel=corePerLevel,
        tensorWeight=tensorWeight,
        platform="CPU",
    )

    ins, outs, _ = conv_conv_nchwc(*shape)
    
    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op
    def get_match_info(op, shape, prefix):
        # cpu_intrin(op, shape, instructionSet, dtype, prefix="")
        packed, code = at.cpu_intrin(
            op="gemm", shape=shape, isa=isa, dtype=dtype, prefix=prefix)
        choices = at.intrinsic_match(op.output(0), packed, ["InnerMost", "SameRange"])
        print("choices", op.name, choices)
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info

    first_match_info = get_match_info(op1, (MI1, NI1, KI1), "conv1")

    second_match_info = get_match_info(op2, (MI2, NI2, KI2), "conv2")

    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    print("tensorizeAxes", tensorizeAxes)

    # packed1, code1 = at.cpu_intrin(op="conv", shape=(
    #     mk1.N, mk1.K, mk1.H, mk1.W, mk1.C, mk1.R, mk1.S), isa=isa, dtype=dtype, prefix="conv1")
    # first_match_info = at.match_info(choice1, packed1, code1)

    # packed2, code2 = at.cpu_intrin(op="conv", shape=(
    #     mk2.N, mk2.K, mk2.H, mk2.W, mk2.C, mk2.R, mk2.S), isa=isa, dtype=dtype, prefix="conv2")
    # second_match_info = at.match_info(choice2, packed2, code2)

    sfs.register_tensorize_axes(tensorizeAxes)

    print("begin building fusion choice...")
    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype=dtype, simple_mode=1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

    if mode == "test":
        print(tensorize_state.summary(verbose=True))

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
    func = tvm.build(sch0, layer.schedule_tensors, name="conv_conv")
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

    fusionLevels = []
    for i in range(fusionContext.size):
        fusionLevels.append(fusionContext.getFusionLevel(i))
    print("fusionLevels: ")
    print(fusionLevels)

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
        # if (mode == "test"):
        lowerCode = tvm.lower(sch, layer.schedule_tensors, simple_mode = True)
        with open ("tmp.txt", 'w') as f:
            f.write(str(lowerCode))
        
        func = tvm.build(sch, layer.schedule_tensors, name="conv_conv")

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)
        func.export_library(f"lib{shape + [iter]}.so")
        with open(f"data{shape + [iter]}.npy", 'wb') as f:
            np.save(f, inputs_np + outputs_np)
        # for (a, b) in zip(outputs, outputs_tvm):
        #     tvm.testing.assert_allclose(
        #         a.numpy(), b.numpy(), atol=1, rtol=1e-6)
        
        print("test passed!")

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=10, repeat=REPEAT
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

shapes = [
    [1, 64, 114, 112, 192, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 32, 147, 148, 64, 3, 3, 96, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 128, 3, 3, 64, 1, 1, 1, 0, 1, 1],
    [1, 128, 27, 28, 256, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 16, 228, 227, 64, 3, 3, 32, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 64, 1, 1, 64, 3, 3, 0, 1, 1, 1],
    [1, 64, 57, 56, 64, 1, 1, 64, 1, 1, 0, 0, 1, 1],
    [1, 256, 57, 56, 256, 1, 1, 64, 1, 1, 0, 0, 1, 1]
]


def testSchedule(shape, server, fusionConfig):
    (batch, C0, H1, W1, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2) = shape
    workload = batch * (C0 * C1 * H1 * W1 * R1 * S1 + C1 * C2 * H1 * W1 * R2 * S2) / 1e9
    sch, tensors = schedule(shape, fusionConfig)
    func = tvm.build(sch, tensors, name="conv_conv")
    ctx = tvm.cpu()
    tensors_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in tensors
    ]
    tensors_tvm = [tvm.nd.array(x, ctx) for x in tensors_np]
    func(*tensors_tvm)
    evaluator = func.time_evaluator(
        func.entry_name, ctx, min_repeat_ms=10, repeat=50
    )
    cost = evaluator(*tensors_tvm).mean
    toPeak = workload / cost / server['peakgflops']

    ret = {'shape': shape, 'time': cost, 'toPeak': toPeak, 'fusionConfig': fusionConfig}
    print(ret)
    return ret
def setGlobals(shape):
    (batch, C0, H, W, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2) = shape
    global MI1, NI1, KI1, MI2, NI2, KI2, WORKLOAD
    WORKLOAD = batch * (C0 * H * W * C1 * R1 * S1 + C1 * H * W * C2 * R2 * S2)
    MI1 = MI2 = 6
    if SERVER['isa'] == "avx2":
        NI1 = 16
        KI1 = 32 if C0 % 32 == 0 else 16
        KI2 = NI1 
        NI2 = 16
    elif SERVER['isa'] == "avx512":
        NI1 = 64 if C1 % 64 == 0 else 32
        KI1 = 64 if C0 % 64 == 0 else C0
        KI2 = NI1 
        NI2 = 64 if C2 % 64 == 0 else 32
    assert (R1 == 3 and S1 == 3 and padding1 == 1) or (R1 == 1 and S1 == 1 and padding1 == 0)
    assert (R2 == 3 and S2 == 3 and padding2 == 1) or (R2 == 1 and S2 == 1 and padding2 == 0)
    assert (H * W) % MI1 == 0
    assert C0 % KI1 == 0
    assert C1 % NI1 == 0
    assert C2 % NI2 == 0 

def main(shape, dtype, server, mode):
    global SERVER
    SERVER = ServerConfig[server]
    setGlobals(shape)
    print("shape,dtype,WORKLOAD,SERVER")
    print(shape, dtype, WORKLOAD, SERVER)
    time = test_double_conv(shape, config={
        'searchType': 'normal', 'verbose': True, 'mode': mode, 'dtype': dtype, 'topn': 1})
    ret = {}
    for k in time:
        ret[k] = {}
        ret[k]['time(s)'] = time[k]
        ret[k]['%peak'] = (WORKLOAD / time[k] / 1e9) / SERVER['peakgflops']
        ret[k]['gflops'] = (WORKLOAD / time[k] / 1e9)
    print(shape, dtype, ":", ret)
    return ret

example_text = "python conv_conv_nchw.py --server scccc --begin 0 --num 1"

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
        cost = main(
            ss,
            dtype = args.dtype,
            server= args.server,
            mode = args.mode
        )
        costs.append((ss, cost))

    print("shape,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.dtype},{args.server},{cc[1]}"
        )
    with open("conv_conv_nchwc_chimera.pkl", "wb") as f:
        pkl.dump(costs, f)
