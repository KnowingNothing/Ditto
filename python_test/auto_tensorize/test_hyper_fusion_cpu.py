from nbformat import write
import pytest
import tvm
import tvm.testing
from tvm import te
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math
import numpy as np
from ditto.hardware.hw_param import hardware_param
import time
import random
import pickle as pkl
import subprocess

MI = 6
NI1 = 16
KI1 = 64
NI2 = 16
KI2 = 16


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


# 1,128,388,388,64,3,3,64,3,3


def ConvReluConv(
    shape,
    padding1=1,
    padding2=1,
    stride1=1,
    stride2=1,
    in_dtype="float32",
    mk1=MicroKernel(),
    mk2=MicroKernel(),
):
    """
            N1 K4 H3 W8 C1 R3 S3
    Conv1   N C1 P0 Q0 C0 R1 S1
    Conv1   N C2 P1 Q1 C1 R2 S2
    """
    assert len(shape) == 10
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2 = shape
    P1 = (P0 + 2 * padding1 - R1) // stride1 + 1
    Q1 = (Q0 + 2 * padding1 - S1) // stride1 + 1
    P2 = (P1 + 2 * padding2 - R2) // stride2 + 1
    Q2 = (Q1 + 2 * padding2 - S2) // stride2 + 1
    assert C1 % mk1.K == 0
    assert C2 % mk2.K == 0
    assert P0 % mk1.H == 0
    assert P1 % mk2.H == 0
    assert Q0 % mk1.W == 0
    assert Q1 % mk2.W == 0
    assert C0 % mk1.C == 0
    assert C1 % mk2.C == 0

    Img = tvm.te.placeholder([N, C0, P0, Q0], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder([C1, C0, R1, S1], dtype=in_dtype, name="Weight1")
    Weight2 = tvm.te.placeholder([C2, C1, R2, S2], dtype=in_dtype, name="Weight2")

    Pad1 = tvm.te.compute(
        [N, C0, P0 + 2 * padding1, Q0 + 2 * padding1],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding1, p < P0 + padding1, q >= padding1, q < Q0 + padding1
            ),
            Img[n, c, p - padding1, q - padding1],
            tvm.tir.const(0, in_dtype),
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
            tvm.tir.const(0, in_dtype),
        ),
        name="relu",
    )

    Pad2 = tvm.te.compute(
        [N, C1, P1 + 2 * padding2, Q1 + 2 * padding2],
        lambda n, c, p, q: tvm.tir.if_then_else(
            tvm.tir.all(
                p >= padding2, p < P0 + padding2, q >= padding2, q < Q0 + padding2
            ),
            Conv1_relu[n, c, p - padding2, q - padding2],
            tvm.tir.const(0, in_dtype),
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
            ]
            * Weight2[ko2 * mk2.K + ki2, co2 * mk2.C + ci2, r2, s2],
            axis=[co2, ci2, r2, s2],
        ),
        name="conv2_fact",
    )
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
        [Conv2]
    )


def BatchGemmReluGemm(
    batch=1, M=516, N=64, K=64, L=512, in_dtype="float32", acc_dtype="float32"
):
    assert M % MI == 0
    assert N % NI2 == 0
    assert K % KI1 == 0
    assert L % NI1 == 0
    assert L % KI2 == 0

    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=in_dtype)

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
            A_shared[b, mo, rko, mi, rki].astype(acc_dtype)
            * B_shared[b, rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    relu = tvm.te.compute(
        [batch, M // MI, L // NI1, MI, NI1],
        lambda b, mo, lo, mi, li: tvm.tir.if_then_else(
            D_frag[b, mo, lo, mi, li] > 0, D_frag[b, mo, lo, mi, li], 0
        ).astype(in_dtype),
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
            relu[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_ext[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, n // NI2, m % MI, n % NI2].astype(in_dtype),
        name="F",
    )

    return (
        [A, B, C],
        [F]
    )

def test_tensorize_cpu():
    ins, outs = BatchGemmReluGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix):
        packed, code = at.cpu_intrin("gemm", shape, dtype = "float32", prefix = prefix)
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info 

    first_match_info = get_match_info(op1, (MI, NI1, KI1), "gemm1")

    second_match_info = get_match_info(op2, (MI, NI2, KI2), "gemm2")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=-1,
        num_groups=20,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=[32 * 16, 32 * 1024, 256 * 1024, 35840 * 1024],
        bandwidth=[293.72, 81.72, 38.54, 13.14],
        coresPerCacheLevel=[1, 1, 1, 10],
        tensorWeight=[1.0,2.0],
        platform="CPU",
    )

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype="float32", simple_mode=-1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )
    print(tensorize_state.summary(verbose=True))

    t0 = time.time()
    tensorize_param = at.build_cpu_tensorize_param(sfs, fuse_choice, CPU, 4)
    t1 = time.time()
    print(f"build param takes {t1-t0} s")

    print(tensorize_param)

    sch = at.tensorize_cpu(layer, tensorize_state, CPU, tensorize_param)

    print(tvm.lower(sch, layer.schedule_tensors, simple_mode=-1))


def test_tensorize_cpu_run(parallelism=1):
    ins, outs = BatchGemmReluGemm()

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix):
        packed, code = at.cpu_intrin("gemm", shape, dtype = "float32", prefix = prefix)
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info 

    first_match_info = get_match_info(op1, (MI, NI1, KI1), "gemm1")

    second_match_info = get_match_info(op2, (MI, NI2, KI2), "gemm2")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=-1,
        num_groups=20,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=[32 * 16, 32 * 1024, 256 * 1024, 35840 * 1024],
        bandwidth=[293.72, 81.72, 38.54, 13.14],
        coresPerCacheLevel=[1, 1, 1, 10],
        tensorWeight=[1.0,2.0],
        platform="CPU",
    )

    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype="float32", simple_mode=-1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )
    print(tensorize_state.summary(verbose=True))

    t0 = time.time()
    tensorize_param = at.build_cpu_tensorize_param(sfs, fuse_choice, CPU, 4)
    t1 = time.time()
    print(f"build param takes {t1-t0} s")

    print(tensorize_param)

    sch = at.tensorize_cpu(layer, tensorize_state, CPU, tensorize_param)

    func = tvm.build(sch, layer.schedule_tensors, name="bmm")

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in outs
    ]
    ctx = tvm.cpu()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(
        func.entry_name, ctx, min_repeat_ms=100, repeat=1000
    )
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    workload = 516 * 512 * 64 * 2
    print(f"Our code uses {cost} ms")
    print(
        f"gflops:{workload / 1e6 / cost}, peak:{parallelism*16*2.2}, ratio: {workload / 1e6 / cost/(parallelism*16*2.2)}"
    )


def test_tensorize_cpu_run_conv(parallelism=1, config={}):
    mk1 = MicroKernel()
    mk2 = MicroKernel()
    shape = (1,128,384,384,64,3,3,64,3,3)
    ins, outs = ConvReluConv(shape, mk1 = mk1, mk2 = mk2)

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix):
        packed, code = at.cpu_intrin("conv", shape, dtype = "float32", prefix = prefix)
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info 

    first_match_info = get_match_info(op1, (mk1.N, mk1.K, mk1.H, mk1.W, mk1.C, mk1.R, mk1.S), "conv1")

    second_match_info = get_match_info(op2, (mk2.N, mk2.K, mk2.H, mk2.W, mk2.C, mk2.R, mk2.S), "conv2")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    cacheSizes = [1024, 32 * 1024, 256 * 1024, 35840 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]

    if "cacheSizes" in config:
        cacheSizes = config["cacheSizes"]
    if "bandwidth" in config:
        bandwidth = config["bandwidth"]
    if "tensorWeight" in config:
        tensorWeight = config["tensorWeight"]

    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=1,
        num_groups=parallelism,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=cacheSizes,
        bandwidth=bandwidth,
        tensorWeight=tensorWeight,
        coresPerCacheLevel=[1, 1, 1, 10],
        platform="CPU",
    )
    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype="float32", simple_mode=-1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

    print(tensorize_state.summary(verbose=True))

    tensorize_param = at.build_cpu_tensorize_param(sfs, fuse_choice, CPU, 4)

    sch = at.tensorize_cpu(layer, tensorize_state, CPU, tensorize_param)

    func = tvm.build(sch, layer.schedule_tensors, name="bmm")

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype("float32")
        for y in outs
    ]
    ctx = tvm.cpu()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(
        func.entry_name, ctx, min_repeat_ms=100, repeat=1000
    )
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    workload = 2 * 1 * 128 * 388 * 388 * 64 * 3 * 3
    print(f"Our code uses {cost} ms")
    print(
        f"gflops:{workload / 1e6 / cost}, peak:{parallelism*16*2.2}, ratio: {workload / 1e6 / cost/(parallelism*16*2.2)}"
    )

ConvData = [
    [1, 32, 114, 112, 32, 3, 3],
    [1, 128, 57, 56, 128, 3, 3],
    [1, 256, 30, 32, 256, 3, 3],
    [1, 512, 15, 16, 512, 3, 3],
    [1, 1024, 9, 8, 1024, 3, 3],
    [1, 64, 57, 56, 64, 3, 3],
    [1, 128, 30, 32, 128, 3, 3],
    [1, 256, 15, 16, 256, 3, 3],
    [1, 512, 9, 8, 512, 3, 3],
    [1, 32, 546, 544, 8, 3, 3],
    [1, 64, 273, 272, 32, 3, 3],
    [1, 128, 138, 136, 64, 3, 3],
    [1, 256, 69, 72, 128, 3, 3],
    [1, 512, 36, 40, 256, 3, 3],
    [1, 1024, 18, 24, 512, 3, 3],
]

GEMMData = [
    (1, 258, 256, 256),
    (1, 2052, 32, 128),
    (1, 1026, 256, 32),
    (1, 132, 1024, 2048),
    (1, 516, 512, 64),
    (1, 36, 1024, 32),
    (1, 50172, 64, 192),
    (1, 786, 512, 2304),
]


def test_model_validation(
    batch, M, N, K, L, instructionSet="avx2", prefix="", config={}
):
    prefix = "_".join([str(_) for _ in [batch, M, N, K, L]])
    print(f"doubleGEMM({batch},{M},{K},{L},{N})")

    cacheSizes = [32 * 16, 32 * 1024, 256 * 1024, 35840 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1

    if "cacheSizes" in config:
        cacheSizes = config["cacheSizes"]
    if "bandwidth" in config:
        bandwidth = config["bandwidth"]
    if "tensorWeight" in config:
        tensorWeight = config["tensorWeight"]
    if "searchType" in config:
        searchType = config["searchType"]
    if "mode" in config:
        mode = config["mode"]
    if "parallelism" in config:
        parallelism = config["parallelism"]
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
        coresPerCacheLevel=[1.0, 1.0, 1.0, 10.0],
        tensorWeight=tensorWeight,
        platform="CPU",
    )

    batch = parallelism
    
    ins, outs = BatchGemmReluGemm(batch, M, N, K, L)

    layer = ac.layer([outs[0].op], inputs=ins)

    sfs = at.build_serial_fusion_state(layer)

    op1, op2 = sfs.first_op, sfs.second_op

    def get_match_info(op, shape, prefix):
        packed, code = at.cpu_intrin("gemm", shape, dtype = "float32", prefix = prefix)
        choices = at.intrinsic_match(op.output(0), packed, ['InnerMost', 'SameRange'])
        choice = choices[0]
        match_info = at.match_info(choice, packed, code)
        return match_info 

    first_match_info = get_match_info(op1, (MI, NI1, KI1), "gemm1")

    second_match_info = get_match_info(op2, (MI, NI2, KI2), "gemm2")
    
    tensorizeAxes = list(first_match_info.axis) + list(second_match_info.axis)
    
    sfs.register_tensorize_axes(tensorizeAxes)
    
    fuse_choice = at.build_fusion_choice(
        sfs, hw_param=CPU, dtype="float32", simple_mode=-1
    )

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {op1: first_match_info, op2: second_match_info}
    )

    data_path = f"/home/CORP.PKUSC.ORG/gulang2020/workspace/Ditto/python_test/auto_tensorize/result/{prefix}gemm_data.txt"
    res_path = f"./result/{prefix}gemm_res.txt"

    print("mode is ", mode)
    fusionContext = at.build_fusion_context(
        sfs,
        layer,
        tensorize_state,
        data_path,
        hw_param=CPU,
        dtype="float32",
        searchType=searchType,
        mode="survey",
    )

    ave_time = 0
    ave_ratioToPeak = 0

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

    metrics = []
    num_valid = 0
    print("candidates number: ", fusionContext.size)
    for iter in range(fusionContext.size):
        print("run", iter)
        sch = tvm.te.create_schedule(outs[0].op)

        sch = fusionContext.run(iter, sch, verbose=False)

        # print(tvm.lower(sch, layer.schedule_tensors, simple_mode = True))

        computation = batch * M * (K * L + L * N)  # fusionContext.getComputation(i)

        occupancy = fusionContext.getOccupancy(iter)

        predCost = fusionContext.getPredCost(iter)

        predCostList = [float(_) for _ in fusionContext.getPredCostList(iter)]

        for _ in range(len(predCostList)):
            predCostList[_] *= bandwidth[_]

        print("predCostList: ", predCostList)

        with open("./result/pred.txt", "a") as f:
            shapeS = [str(_) for _ in [batch, M, N, K, L]]
            predListS = [str(_) for _ in predCostList]
            f.write(" ".join(shapeS + predListS))
            f.write("\n")

        func = tvm.build(sch, layer.schedule_tensors, name="bmm")

        data = inputs_np + outputs_np

        filename = f"mat{iter}.npz"

        modulename = f"myfunc_pack{iter}.so"

        np.savez(filename, *data)

        func.export_library(modulename)

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)

        cmd1 = (
            f"python ./runCommand.py python ./profileScript.py {filename} {modulename}"
        )

        cmd2 = "python ./parseAndDump.py"

        p1 = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)

        p1.wait()

        p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout)

        p2.wait()

        # for (a, b) in zip(outputs, outputs_tvm):
        #     tvm.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-3, rtol=1)
        num_valid += 1

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=100, repeat=10
        )

        cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        ave_time += cost
        ratioToPeak = computation / 1e6 / cost / 35.2 / parallelism
        ave_ratioToPeak += ratioToPeak
        print("shape", batch, M, N, K, L, "ratioToPeak:", ratioToPeak)
        metrics.append(
            {
                "time": cost,
                "computation": computation,
                "toPeak": ratioToPeak,
                "predCost": predCost,
                "occupancy": occupancy,
            }
        )
        if mode == "best":
            break
        with open(res_path, "a") as f:
            f.write(f"{iter} {ratioToPeak} {1 / ratioToPeak}\n")

    ave_time /= num_valid
    ave_ratioToPeak /= num_valid
    print("invalid rate: ", 1 - (num_valid / fusionContext.size))
    print("ave_time: ", ave_time)
    print("ave_ratioToPeak: ", ave_ratioToPeak)
    return ave_time, ave_ratioToPeak, data_path, res_path, metrics


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


index = 0


def rank_acc(param, prefix="", writeResult=False):
    cacheSizes, bandwidth, tensorWeight = param[:3], param[3:6], param[6:]
    cacheSizes = [32 * 16] + cacheSizes
    bandwidth = [293.0] + bandwidth
    x, y = [], []
    cost = 0
    valid = 0
    config = {
        "cacheSizes": cacheSizes,
        "bandwidth": bandwidth,
        "tensorWeight": tensorWeight,
        "searchType": "stochastic",
        "mode": "survey",
    }
    for shape in GEMMData:
        B, M, L, K = shape
        print(f"testing shape: doubleGEMM(M={M},N={K},K={K},L={L})")
        prefix = f"rank_acc_{index}"
        try:
            _, _, data_path, res_path, metrics = test_model_validation(
                batch=B,
                M=M,
                N=K,
                K=K,
                L=L,
                instructionSet="avx2",
                prefix=prefix,
                config=config,
            )
            valid += 1
        except:
            print(f"data {shape} error")
            continue
        x = [i["predCost"] for i in metrics]
        y = [i["time"] for i in metrics]
        rank_accuracy = cal_rank_acc(x, y)
        cost += rank_accuracy
        print(f"doubleGEMM(M={M},N={K},K={K},L={L}): rank_acc{rank_accuracy}")
        if writeResult:
            with open(f"doubleGEMM(M={M},N={K},K={K},L={L}).pkl", "wb") as f:
                pkl.dump(metrics, f)

    cost /= valid
    tmp = ["%.2f" % _ for _ in param]
    print("param: ", " ".join(tmp), "cost: ", cost)
    with open("./result/performance.txt", "a") as f:
        f.write("param: ")
        f.write(" ".join(tmp))
        f.write("cost: ")
        f.write("%.4f" % cost)
        f.write("\n")
    return cost


def time_cost(param, parallelism=1, prefix=""):
    cacheSizes, bandwidth, tensorWeight = param[:3], param[3:6], param[6:]
    cacheSizes = [32 * 16] + cacheSizes
    bandwidth = [293.0] + bandwidth
    ave_cost = 0
    print(cacheSizes, bandwidth, tensorWeight)
    config = {
        "cacheSizes": cacheSizes,
        "bandwidth": bandwidth,
        "tensorWeight": tensorWeight,
        "searchType": "normal",
        "mode": "survey",
        "parallelism": parallelism,
    }
    doubleGEMM_valid = 0
    doubleGEMM_cost = 0
    for shape in GEMMData:
        B, M, L, K = shape
        # try:
        time, cost, _, _, _ = test_model_validation(
            batch=B, M=M, N=K, K=K, L=L, instructionSet="avx2", config=config
        )
        ave_cost += cost
        doubleGEMM_cost += cost
        doubleGEMM_valid += 1
        print("shape: ", shape, "time: ", time)
        # except:
        #     print(f"error with data {shape}")
    doubleGEMM_cost /= doubleGEMM_valid
    print(f"double gemm cost is {doubleGEMM_cost}")

    tmp = ["%.2f" % _ for _ in param]
    print("param: ", " ".join(tmp), "cost: ", ave_cost)
    return ave_cost


def sas(costType="time", problem="GEMM"):
    # 20077.01 262144.00 150453156.18 188.55 245.84 1.90 1.60 10.92
    cacheSizes_ = [20077, 256 * 1024, 150453156]
    bandwidth_ = [188.55, 245.84, 13]
    tensorWeight_ = [1.0, 2.0]
    param = cacheSizes_ + bandwidth_ + tensorWeight_
    performance = []
    for _ in range(len(ConvData) + len(GEMMData)):
        performance.append([])

    if costType == "time":
        cost_func = time_cost
    elif costType == "rank_acc":
        cost_func = rank_acc
    E = cost_func(param=param)
    best_param = param
    best_E = 0
    for k in np.arange(0, 1, 0.05):
        T = 1 - k
        new_param = param.copy()
        i = random.randint(0, len(param) - 1)
        for fac in [1 - math.sqrt(T) * 0.5, 1 / (1 - math.sqrt(T) * 0.5)]:
            param_ = param.copy()
            param_[i] *= fac
            E_ = cost_func(param=param_)
            if math.exp(-(E - E_) / T) > random.random():
                new_param = param_.copy()
                E = E_
            if best_E < E_:
                best_E = E_
                best_param = param_.copy()
        param = new_param
        print(
            "param: ",
            param,
        )
    with open("./result/param.txt", "a") as f:
        f.write(str(best_E))
        f.write(" ")
        f.write(" ".join([str(i) for i in best_param]))
        f.write("\n")
    with open("./result/performace.txt", "a") as f:
        for line in performance:
            f.write(" ".join(line))
            f.write("\n")


def setGlobals(instructionSet="avx2"):
    global MI, NI1, KI1, NI2, KI2
    if instructionSet == "avx2":
        MI = 6
        NI1 = 16
        KI1 = 32
        KI2 = 16
        NI2 = 16
    elif instructionSet == "avx512":
        MI = 6


if __name__ == "__main__":
    setGlobals("avx2")
    test_tensorize_cpu()
    test_tensorize_cpu_run()
    test_tensorize_cpu_run_conv()
    test_model_validation(
        1,
        2052,
        128,
        128,
        32,
        "avx2",
        config={"searchType": "stochastic", "mode": "best", "parallelism": 1},
    )
