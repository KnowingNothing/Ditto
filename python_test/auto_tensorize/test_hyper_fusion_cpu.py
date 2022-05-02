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
import torch

MI = 6
NI1 = 16
KI1 = 64
NI2 = 16
KI2 = 16

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

def profile(data, func, surfix):
    filename = f"./data/mat{surfix}.npz"

    modulename = f"./data/myfunc_pack{surfix}.so"

    np.savez(filename, *data)

    func.export_library(modulename)

    cmd1 = (
        f"python ./runCommand.py python ./profileScript.py {filename} {modulename}"
    )

    cmd2 = "python ./parseAndDump.py"

    p1 = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)

    p1.wait()

    p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout)

    p2.wait()

def test_shape(batch, M, N, K, L):
    num = 100
    per = 100
    num = ((num-1) // per + 1) * per
    cost = 0
    for outest in range(num // 100 + 1):
        Q_np = []
        K_np = []
        V_np = []
        Qtensor = []
        Ktensor = []
        Vtensor = []
        QKV_np = []
        for trial in range(per):
            Q_np.append(np.random.uniform(
                size=(batch, M, K)).astype(np.float32))
            K_np.append(np.random.uniform(
                size=(batch, K, L)).astype(np.float32))
            V_np.append(np.random.uniform(
                size=(batch, L, N)).astype(np.float32))
            Qtensor.append(torch.from_numpy(Q_np[trial]))
            Ktensor.append(torch.from_numpy(K_np[trial]))
            Vtensor.append(torch.from_numpy(V_np[trial]))
            QKV_np.append(np.random.uniform(
                size=(batch, M, N)).astype(np.float32))
            for i in range(batch):
                QKV_np[trial][i] = Q_np[trial][i].dot(
                    K_np[trial][i]).dot(V_np[trial][i])

        QKV = []

        start = time.time()
        for trial in range(per):
            QK = torch.bmm(Qtensor[trial], Ktensor[trial])
            # QK_relu = torch.nn.ReLU()(QK)
            QKV.append(torch.bmm(QK, Vtensor[trial]))

        end = time.time()
        cost_tmp = (end - start) / per
        if outest > 0:
            cost += cost_tmp
#        print("%d: %g" % (outest, cost_tmp))

        for trial in range(per):
            np.testing.assert_allclose(
                QKV_np[trial], QKV[trial].numpy(), rtol=1e-3)

    cost /= num // per
    print("averge cost: %g" % cost)

    return cost


def test_model_validation(
    batch, M, N, K, L, instructionSet="avx2", prefix="", config={}
):
    prefix = "_".join([str(_) for _ in [batch, M, N, K, L]])
    print(f"doubleGEMM({batch},{M},{N},{K},{L})")

    cacheSizes = [32 * 16, 32 * 1024, 256 * 1024, 35840 * 1024]
    bandwidth = [293.72, 81.72, 1000.54, 1000.14]
    tensorWeight = [1.0, 2.0]
    searchType = "stochastic"
    mode = "survey"
    parallelism = 1
    verbose = False

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
    if "verbose" in config:
        verbose = config['verbose']
    
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
        mode=mode,
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
    predictedTop = -1
    for iter in range(fusionContext.size):
        print("run", iter)
        sch = tvm.te.create_schedule(outs[0].op)

        sch = fusionContext.run(iter, sch, verbose=verbose)

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

        profile(data, func, str(iter))

        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        func(*inputs_tvm, *outputs_tvm)

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
        with open(res_path, "a") as f:
            f.write(f"{iter} {ratioToPeak} {1 / ratioToPeak}\n")
        if (predictedTop < 0):
            predictedTop = ratioToPeak

    ave_time /= num_valid
    ave_ratioToPeak /= num_valid
    print("invalid rate: ", 1 - (num_valid / fusionContext.size))
    print("ave_time: ", ave_time)
    print("ave_ratioToPeak: ", ave_ratioToPeak)
    return ave_time, ave_ratioToPeak, data_path, res_path, metrics, predictedTop





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

def generateConvShapes():
    B = [1, 4, 8, 16]
    O = [64, 128, 256, 512, 1024, 2048]
    b_i = random.randint(0, len(B))
    shape = [B[b_i]]
    for _ in range(4):
        shape.append(O[random.randint(0, len(O))])
    shape[1] = (shape[1] + 5) // 6 * 6
    return shape

def performance_test_script():
    torch.set_num_threads(20)
    shape = generateConvShapes()
    _, _, _, _, _, ratioToPeak = test_model_validation(*shape, config = {'searchType': 'normal', 'mode': 'survey', 'parallel': 20})
    ratioToPeak_torch = test_shape(*shape)
    with open("./result/performance_validation.txt", "a") as f:
        data = shape + [ratioToPeak, ratioToPeak_torch, ratioToPeak / ratioToPeak_torch] 
        data = [float(_) for _ in data]
        f.write(' '.join(data) + '\n')
    cmd1 = (
        f"python ./runCommand.py python ./test_torch {shape[0]} {shape[1]} {shape[2]} {shape[3]} {shape[4]}"
    )

    cmd2 = "python ./parseAndDump_torch.py"

    p1 = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)

    p1.wait()

    p2 = subprocess.Popen(cmd2.split(), stdin=p1.stdout)

    p2.wait()

if __name__ == "__main__":
    setGlobals("avx2")
    # test_tensorize_cpu()
    # test_tensorize_cpu_run()
    # test_tensorize_cpu_run_conv()
    test_model_validation(
        20,
        516,
        64,
        64,
        512,
        "avx2",
        config={"searchType": "normal", "mode": "survey", "parallelism": 1, "verbose": False},
    )
