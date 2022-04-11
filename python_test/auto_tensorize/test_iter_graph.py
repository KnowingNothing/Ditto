import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto.hardware.hw_param import hardware_param
import math

MI = NI = KI = 16


def BatchGemmSoftmaxGemm(
    batch=12, M=512, N=64, K=64, L=512, in_dtype="float16", acc_dtype="float32"
):
    assert M % MI == 0
    assert N % NI == 0
    assert K % KI == 0
    assert L % NI == 0
    assert L % KI == 0

    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)
    C = tvm.te.placeholder([batch, L, N], name="C", dtype=in_dtype)

    A_shared = tvm.te.compute(
        [batch, M // MI, K // KI, MI, KI],
        lambda b, mo, ko, mi, ki: A[b, mo * MI + mi, ko * KI + ki],
        name="A_shared",
    )

    B_shared = tvm.te.compute(
        [batch, K // KI, L // NI, KI, NI],
        lambda b, ko, lo, ki, li: B[b, ko * KI + ki, lo * NI + li],
        name="B_shared",
    )

    rko = tvm.te.reduce_axis([0, K // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    D_frag = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A_shared[b, mo, rko, mi, rki].astype(acc_dtype)
            * B_shared[b, rko, lo, rki, li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="D_frag",
    )

    exp = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.exp(D_frag[b, mo, lo, mi, li]).astype(
            in_dtype
        ),
        name="exp",
    )

    ext_N = 2 ** math.ceil(math.log2(N // NI + 1)) * NI
    C_ext = tvm.te.compute(
        [batch, L // KI, ext_N // NI, KI, NI],
        lambda b, lo, no, li, ni: tvm.tir.if_then_else(
            no * NI + ni < N,
            C[b, lo * NI + li, no * NI + ni],
            tvm.tir.const(1, in_dtype),
        ),
        name="C_ext",
    )

    rlo = tvm.te.reduce_axis([0, L // KI], "rlo")
    rli = tvm.te.reduce_axis([0, KI], "rli")
    E_frag = tvm.te.compute(
        [batch, M // MI, ext_N // NI, MI, NI],
        lambda b, mo, no, mi, ni: tvm.te.sum(
            exp[b, mo, rlo, mi, rli].astype(acc_dtype)
            * C_ext[b, rlo, no, rli, ni].astype(acc_dtype),
            axis=[rlo, rli],
        ),
        name="E_frag",
    )

    F = tvm.te.compute(
        [batch, M, N],
        lambda b, m, n: E_frag[b, m // MI, n // NI, m % MI, n % NI].astype(in_dtype)
        / (
            E_frag[b, m // MI, ext_N // NI - 1, m % MI, NI - 1].astype(in_dtype)
            + tvm.tir.const(1e-5, in_dtype)
        ),
        name="F",
    )

    return (
        [A, B, C],
        [F],
        [
            D_frag.op.axis[-2],
            D_frag.op.axis[-1],
            rki,
            E_frag.op.axis[-1],
            E_frag.op.axis[-2],
            rli,
        ],
    )


MI1  = 6
NI1 = 64
KI1 = 64

MI2  = 6
NI2 = 64
KI2 = 64

def ceil(a, b):
    return ((a + b - 1) // b)

def conv_relu_conv_nchwc(
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

    print("CI, CO, PQ1O, PQ1I, K1I, K1O, PQ2I, PQ2O, K2I, K2O, RK1I, RK1O")
    print(CI, CO, PQ1O, PQ1I, K1I, K1O, PQ2I, PQ2O, K2I, K2O, RK1I, RK1O)


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

    relu1 = tvm.te.compute(
        [N, K1O, PQ1O, PQ1I, K1I],
        lambda n, ko, pqo, pqi, ki: tvm.tir.if_then_else(
            conv1_frag[n, ko, pqo, pqi, ki] > tvm.tir.const(0, acc_dtype),
            conv1_frag[n, ko, pqo, pqi, ki].astype(in_dtype),
            tvm.tir.const(0, in_dtype),
        ),
        name="relu1",
    )

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
            relu1[
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
    return [Img, Weight1, Weight2], [conv2_frag],choice1+choice2

def test_iter_graph(path=""):
    # [1, 64, 57, 56, 128, 1, 1, 64, 1, 1, 0, 0, 1, 1]
    # B = 12
    # M = 512
    # N = 64
    # K = 64
    # L = 512
    # # ins, outs = GemmReLUGemm(M, N, K, L)
    # ins, outs, tensorizeAxes = BatchGemmSoftmaxGemm(B, M, N, K, L)
    ins, outs, tensorizeAxes =\
     conv_relu_conv_nchwc(1, 64, 57, 56, 128, 1, 1, 64, 1, 1, 0, 0, 1, 1)
    A, B, E = ins
    (F,) = outs
    layer = ac.layer(F.op, inputs=[A, B, E])
    # print(layer)
    sfs = at.build_serial_fusion_state(layer)   
    sfs.register_tensorize_axes(tensorizeAxes)
    # print(sfs)
    ig = at.build_iter_graph(sfs)
    print(ig)
    # print(ig)
    return ig

def test_schedule_independency():
    ig = test_iter_graph()
    ig.set_first_op_tiling([1, 8, 16, 4])
    ig.set_first_op_permute([0, 1, 2, 3])
    ig.set_second_op_tiling([1, 32, 8, 4])
    ig.set_second_op_permute([0, 1, 2, 3])
    ig.set_attach(2)
    ig.apply_all()
    ig.display()

def test_workflow():
    ig = test_iter_graph()
    print(ig)
    ig.set_first_op_permute([2, 1, 0,3])
    ig.set_first_op_tiling([1, 8, 16, 4])
    ig.set_second_op_tiling([1, 32, 8, 4])
    ig.set_second_op_permute([0, 1, 2, 3])
    ig.set_attach(2)
    ig.apply_all()
    ig.display()
    # test reschedule and reapply_all
    ig.set_first_op_permute([0, 1, 2,3])
    ig.set_first_op_tiling([8, 16, 32, 2])
    ig.set_second_op_tiling([12, 16, 4, 1])
    ig.set_second_op_permute([0, 1, 3, 2])
    ig.set_attach(2)
    ig.apply_all()
    ig.display()


# test schedule independency

def test_analyse():
    ig = test_iter_graph()
    print(ig)
    it = at.build_fusion_item(
        firstOpTiling = [1, 8, 16, 4],
        secondOpTiling = [1, 32, 8, 4],
        firstOpPermute = [0, 1, 2, 3],
        secondOpPermute = [0, 1, 2, 3],
        attachPos = 2,
        fusionLevel = 2,
    )
    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]
    parallelism = 20
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
        cacheSizes=cacheSizes,
        bandwidth=bandwidth,
        coresPerCacheLevel=[1.0, 1.0, 1.0, 16.0],
        tensorWeight=tensorWeight,
        platform="CPU",
    )
    ig.set_config(hw_param = CPU, dtype = "float32")
    ig.set_fusion(it)
    res = ig.analyse()
    print(res)
    ig.display()
    log = res.getLog()
    print(log)

def test_search():
    ig = test_iter_graph()
    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]
    parallelism = 20
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
    searchDriver = at.build_search_driver(
        ig, ["time", "static analysis"], "bruteForce", CPU, "float16"
    )
    print(searchDriver)
    fusionSpace = searchDriver.get_fusion_space()
    print(fusionSpace)
    fusionSpace.set_first_op_tiling_mandatory([16, 16, 4])
    fusionSpace.set_second_op_tiling_mandatory([16, 4, 4])
    print(fusionSpace)
    # the fusionItem
    print("begin search")
    it = searchDriver.search()
    print("end search")
    print(it)
    if it:
        ig.set_fusion(it)
        res = ig.analyse()
        ig.display()
        print(res.getLog())


def test_templates():
    ig = test_iter_graph()
    cacheSizes = [32 * 16, 32 * 1024, 1024 * 1024, 11264 * 1024]
    bandwidth = [293.72, 81.72, 38.54, 13.14]
    tensorWeight = [1.0, 2.0]
    parallelism = 20
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
    searchDriver = at.build_search_driver(
        ig, ["time", "static analysis"], "bruteForce", CPU, "float16"
    )
    fusionSpace = searchDriver.get_fusion_space()
    fusionSpace.set_first_op_permute_mandatory([[0, 1, 2]])
    fusionSpace.set_first_op_tiling_mandatory([1, 1, 1])
    templates = {
        "F1": ([0, 1, 2], 3, [-1, -1, -1]),
        "F2": ([0, 1, 2], 2, [-1, -1, 1]),
        "F3": ([0, 2, 1], 2, [-1, 1, -1]),
        "F4": ([1, 2, 0], 2, [1, -1, -1]),
        "F5": ([0, 1, 2], 1, [-1, 1, 1]),
        "F6": ([1, 0, 2], 1, [1, -1, 1]),
        "F7": ([2, 0, 1], 1, [1, 1, -1]),
    }
    res = {}
    for name, template in templates:
        fusionSpace.set_second_op_permute_mandatory([template[0]])
        fusionSpace.set_attach_mandatory([template[1]])
        fusionSpace.set_second_op_tiling_mandatory(template[2])
        print(fusionSpace)
        it = searchDriver.search()
        res[name] = it
    print(res)

if __name__ == "__main__":
    test_iter_graph()
    #test_schedule_independency()
    #test_workflow()
    #test_analyse()
    # test_search()
    # test_templates()
