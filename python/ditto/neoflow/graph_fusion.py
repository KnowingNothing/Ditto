from ditto import auto_compute as ac
from ditto import autograd as ag
from ditto import hardware as hw


def default_fusion_cost_func(to_fuse, fused, hw_param):
    """
    bandwidth: GB/s
    peak_perf: GFLOPs
    launch_latency: (s)
    """
    bytes_of_dtype = 4
    bandwidth = hw_param.dram_bandwidth
    peak_perf = hw_param.f32_peak_perf
    launch_latency = hw_param.launch_latency
    to_fuse_grad = []
    original_data_transfer = 0
    original_gflops = 0
    for layer in to_fuse:
        grad_layer = ag.grad_layer(layer)
        to_fuse_grad.append(grad_layer)
        original_data_transfer += layer.data_transfer_amount
        original_data_transfer += grad_layer.data_transfer_amount
        original_gflops += layer.gflops
        original_gflops += grad_layer.gflops
    
    fused_grad = ag.grad_layer(fused)
    new_data_transfer = fused.data_transfer_amount + fused_grad.data_transfer_amount
    new_gflops = fused.gflops + fused_grad.gflops
    
    R = (original_data_transfer - new_data_transfer) * bytes_of_dtype / 1e9 / bandwidth + (original_gflops - new_gflops) / peak_perf + (len(to_fuse) * 2 - 2) * launch_latency
    return R


def graph_fusion(graph, cost_func=default_fusion_cost_func, hw_param=hw.V100):
    state = ac.create_graph_state(graph)
    
    all_layers = []
    for layer in graph.all_layers:
        layers = state.normalize_partition_layer(layer)
        # print(layer, flush=True)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", flush=True)
        # for l in layers:
        #     print(l, flush=True)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", flush=True)
        # print()
        all_layers.extend(layers)
        
    tail = 0
    num_layers = len(all_layers)
    assert num_layers > 0
    cur_layer = all_layers[0]
    fused_layers = []
    while tail < num_layers:
        to_fuse = ac.find_convex_layers(cur_layer, all_layers[tail])
        if len(to_fuse) > 0:
            fused = state.fuse_layer(cur_layer, all_layers[tail], modify=False)
            if cost_func(to_fuse, fused, hw_param):
                fused = state.fuse_layer(cur_layer, all_layers[tail], modify=True)
                fused_layers.append(fused)
                cur_layer = fused
            else:
                cur_layer = all_layers[tail]
        else:
            cur_layer = all_layers[tail]
        tail += 1

    # for fused in fused_layers:
    #     print()
    #     print("*******************************", flush=True)
    #     print(fused, flush=True)
    #     print("*******************************", flush=True)
        
    # for layer in state.get_current_layers():
    #     print()
    #     print("+++++++++++++++++++++++++++++++++", flush=True)
    #     print(layer, flush=True)
    #     print("+++++++++++++++++++++++++++++++++", flush=True)
    return state.make_compute(graph.graph_inputs)