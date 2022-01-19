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

def test_iter_graph():
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
    ig = at.build_iter_graph(sfs)
    # print(ig)
    return ig

# test workflow
# ig = test_iter_graph()
# ig.set_first_op_permute([2,1,0])
# ig.set_first_op_tiling([8,16,32])
# ig.set_second_op_tiling([4,5,6])
# ig.set_second_op_permute([0,2,1])
# ig.set_fusion(2)
# ig.synchronize()
# # test reschedule and resynchronize
# ig.set_first_op_permute([0,1,2])
# ig.set_first_op_tiling([8,16,32])
# ig.set_second_op_tiling([32, 16, 4])
# ig.set_second_op_permute([0,1,2])
# ig.set_fusion(2)
# ig.synchronize()

# test schedule independency
ig = test_iter_graph()
ig.set_first_op_tiling([8,16,32])
ig.set_first_op_permute([0,1,2])
ig.set_second_op_tiling([32, 16, 4])
ig.set_second_op_permute([0,1,2])
ig.set_attach(2)
ig.synchronize()

# test analysis

hp = hardware_param(
    256/4, # register_per_processor_kb: float,
    96, # shared_memory_per_group_kb: float, up to 96KB
    12080, # shared_memory_bandwidth_gbs: float,
    16, # global_memory_gb: float,
    750, # global_memory_bandwidth_gbs: float,
    4, # num_processors_per_group: int,
    80, # num_groups: int,
    14*1e3, # fp32_peak_perf_gflops: float,
    5*1e-6, # launch_latency_s: float
)

featureLog = at.build_feature_log(ig, hp)
print(ig)
print(featureLog)

searchDriver = at.build_search_driver(ig, ["time", "static analysis"], "bruteForce")
print(searchDriver)
fusionSpace = searchDriver.get_fusion_space()
print(fusionSpace)
fusionSpace.set_first_op_tiling_mandatories([[4,8,12], [8,64,64]])
fusionSpace.set_second_op_tiling_mandatories([[28,8,12], [8,64,64]])
print(searchDriver)
# the fusionItem
it = searchDriver.search()
print(it)
# the fusionResult
res = searchDriver.eval(it)
print(res)
# the visulize
ig.set_schedule(it)
ig.display()
print(ig)

# raw method for building fusionitem
it = at.build_fusion_item([8,16,32], [32, 16, 4], [0,2,1], [0,1,2], 2)
print(it)
ig.set_schedule(it)
ig.display()
