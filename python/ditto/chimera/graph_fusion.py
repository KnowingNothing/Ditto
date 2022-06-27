from ditto import auto_compute as ac


def graph_fusion(
    graph,
):
    state = ac.create_graph_state(graph)

    all_layers = [x for x in graph.all_layers]

    tail = 0
    next_id = 0
    num_layers = len(all_layers)
    assert num_layers > 0
    cur_layer = all_layers[0]
    fused_layers = []
    while tail < num_layers:
        cur_layer = all_layers[next_id]
        next_id += 1
        if next_id > tail:
            tail = next_id
        if tail >= num_layers:
            break
        if ac.nn.module.is_self_attention_layer(cur_layer):
            cur_layer = all_layers[next_id]
            next_id += 1
            if next_id > tail:
                tail = next_id
            continue
        if ac.nn.module.is_act_layer(all_layers[tail]) or ac.nn.module.is_elem_layer(
            all_layers[tail]
        ):
            to_fuse = ac.find_convex_layers(cur_layer, all_layers[tail])
            if len(to_fuse) > 0:
                fused = state.fuse_layer(cur_layer, all_layers[tail], modify=True)
                fused_layers.append(fused)
                cur_layer = fused
                next_id = tail + 1
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
