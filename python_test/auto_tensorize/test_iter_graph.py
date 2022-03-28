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



def test_iter_graph(path=""):
    B = 12
    M = 512
    N = 64
    K = 64
    L = 512
    # ins, outs = GemmReLUGemm(M, N, K, L)
    ins, outs, tensorizeAxes = BatchGemmSoftmaxGemm(B, M, N, K, L)
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
    # test_iter_graph()
    # test_schedule_independency()
    # test_workflow()
    # test_analyse()
    test_search()
    # test_templates()
