import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto.hardware.hw_param import hardware_param


def GemmReLUGemm(M, N, K, L):
    A = tvm.te.placeholder([M, K], name="A", dtype="float32")
    B = tvm.te.placeholder([K, L], name="B", dtype="float32")
    E = tvm.te.placeholder([L, N], name="E", dtype="float32")
    k = tvm.te.reduce_axis([0, K], "rk")
    C = tvm.te.compute(
        [M, L],
        lambda m1, l1:
            tvm.te.sum(A[m1, k] * B[k, l1], axis=k),
        name="C"
    )
    D = tvm.te.compute(
        [M, L],
        lambda i, j:
            tvm.tir.if_then_else(
                C[i, j] > 0,
                C[i, j],
                0.0
        ),
        name="D"
    )
    l = tvm.te.reduce_axis([0, L], "rl")
    F = tvm.te.compute(
        [M, N],
        lambda m2, n2:
            tvm.te.sum(
                D[m2, l] * E[l, n2], axis=l
        ),
        name="F"
    )
    return [A, B, E], [F]


def test_iter_graph(path=""):
    M = 512
    N = 64
    K = 64
    L = 512
    ins, outs = GemmReLUGemm(M, N, K, L)
    A, B, E = ins
    F, = outs
    layer = ac.layer(F.op, inputs=[A, B, E])
    # print(layer)
    sfs = at.build_serial_fusion_state(layer)
    # print(sfs)
    ig = at.build_iter_graph(sfs, path)
    print(ig)
    # print(ig)
    return ig


def test_workflow():
    ig = test_iter_graph()
    ig.set_first_op_permute([2, 1, 0])
    ig.set_first_op_tiling([8, 16, 32])
    ig.set_second_op_tiling([4, 5, 6])
    ig.set_second_op_permute([0, 2, 1])
    ig.set_attach(2)
    ig.apply_all()
    # test reschedule and reapply_all
    ig.set_first_op_permute([0, 1, 2])
    ig.set_first_op_tiling([8, 16, 32])
    ig.set_second_op_tiling([32, 16, 4])
    ig.set_second_op_permute([0, 1, 2])
    ig.set_attach(2)
    ig.apply_all()

# test schedule independency


def test_schedule_independency():
    ig = test_iter_graph()
    ig.set_first_op_tiling([8, 16, 32])
    ig.set_first_op_permute([0, 1, 2])
    ig.set_second_op_tiling([32, 16, 4])
    ig.set_second_op_permute([0, 1, 2])
    ig.set_attach(2)
    ig.apply_all()


def test_analyse():
    ig = test_iter_graph()
    print(ig)
    it = at.build_fusion_item([16, 16, 4], [16, 4, 4], [0, 2, 1], [0, 2, 1], 2)
    ig.set_fusion(it)
    hp = hardware_param(
        256/4,  # register_per_processor_kb: float,
        96,  # shared_memory_per_group_kb: float, up to 96KB
        12080,  # shared_memory_bandwidth_gbs: float,
        16,  # global_memory_gb: float,
        750,  # global_memory_bandwidth_gbs: float,
        4,  # num_processors_per_group: int,
        80,  # num_groups: int,
        14*1e3,  # fp32_peak_perf_gflops: float,
        5*1e-6,  # launch_latency_s: float
    )
    res = ig.analyse(hp, 4, 1)
    print(res)
    ig.display()
    log = res.getLog()
    print(log)

# test search


def test_search():
    ig = test_iter_graph()
    hp = hardware_param(
        256/4,  # register_per_processor_kb: float,
        96,  # shared_memory_per_group_kb: float, up to 96KB
        12080,  # shared_memory_bandwidth_gbs: float,
        16,  # global_memory_gb: float,
        750,  # global_memory_bandwidth_gbs: float,
        4,  # num_processors_per_group: int,
        80,  # num_groups: int,
        14*1e3,  # fp32_peak_perf_gflops: float,
        5*1e-6,  # launch_latency_s: float
    )

    # featureLog = at.build_feature_log(ig, hp)
    # print(ig)
    # print(featureLog)

    searchDriver = at.build_search_driver(
        ig, ["time", "static analysis"], "bruteForce", hp, "float16")
    print(searchDriver)
    fusionSpace = searchDriver.get_fusion_space()
    print(fusionSpace)
    fusionSpace.set_first_op_tiling_mandatory([16, 16, 4])
    fusionSpace.set_second_op_tiling_mandatory([16, 4, 4])
    print(fusionSpace)
    # the fusionItem
    it = searchDriver.search()
    print(it)
    if it:
        ig.set_fusion(it)
        res = ig.analyse(hp, 2, 1)
        ig.display()
        print(res.getLog())


def test_templates():
    ig = test_iter_graph()
    hp = hardware_param(
        256/4,  # register_per_processor_kb: float,
        96,  # shared_memory_per_group_kb: float, up to 96KB
        12080,  # shared_memory_bandwidth_gbs: float,
        16,  # global_memory_gb: float,
        750,  # global_memory_bandwidth_gbs: float,
        4,  # num_processors_per_group: int,
        80,  # num_groups: int,
        14*1e3,  # fp32_peak_perf_gflops: float,
        5*1e-6,  # launch_latency_s: float
    )


    searchDriver = at.build_search_driver(
        ig, ["time", "static analysis"], "bruteForce", hp, "float32")
    fusionSpace = searchDriver.get_fusion_space()
    fusionSpace.set_first_op_permute_mandatory([[0, 1, 2]])
    fusionSpace.set_first_op_tiling_mandatory([1, 1, 1])
    templates = {
        'F1': ([0, 1, 2], 3, [-1, -1, -1]),
        'F2': ([0, 1, 2], 2, [-1, -1, 1]),
        'F3': ([0, 2, 1], 2, [-1, 1, -1]),
        'F4': ([1, 2, 0], 2, [1, -1, -1]),
        'F5': ([0, 1, 2], 1, [-1, 1, 1]),
        'F6': ([1, 0, 2], 1, [1, -1, 1]),
        'F7': ([2, 0, 1], 1, [1, 1, -1])
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


def test_write_file():
    ig = test_iter_graph("./tmp_.txt")
    print(ig)
    it = at.build_fusion_item([16, 16, 4], [16, 4, 4], [0, 2, 1], [0, 2, 1], 2)
    ig.set_fusion(it)
    hp = hardware_param(
        256/4,  # register_per_processor_kb: float,
        96,  # shared_memory_per_group_kb: float, up to 96KB
        12080,  # shared_memory_bandwidth_gbs: float,
        16,  # global_memory_gb: float,
        750,  # global_memory_bandwidth_gbs: float,
        4,  # num_processors_per_group: int,
        80,  # num_groups: int,
        14*1e3,  # fp32_peak_perf_gflops: float,
        5*1e-6,  # launch_latency_s: float
    )
    res = ig.analyse(hp, 4, 1)
    ig.display()
    log = res.getLog()


if __name__ == '__main__':
    test_iter_graph()
    test_schedule_independency()
    test_workflow()
    test_analyse()
    test_search()
    test_templates()
    test_write_file()
