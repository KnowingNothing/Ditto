from ditto import auto_compute as ac
from collections import OrderedDict


def extract_tasks_from_model(inputs, fget_model):
    """
    Get tasks from model

    Argas:
    ---
    inptus: List[tvm.te.Tensor]

    fget_model: callable
        function to get the model

    Returns:
    ---
    Dict{workload_key(str):List[ditto.auto_compute.Layer]}
    """
    model = fget_model()
    outputs = model(*inputs)

    graph = ac.graph(inputs, outputs)
    all_layers = graph.all_layers
    tasks = OrderedDict()
    for layer in all_layers:
        if layer.fingerprint not in tasks:
            tasks[layer.fingerprint] = [layer]
        else:
            tasks[layer.fingerprint].append(layer)
    return tasks


def extract_tasks_from_graph(graph):
    all_layers = graph.all_layers
    tasks = OrderedDict()
    for layer in all_layers:
        if layer.fingerprint not in tasks:
            tasks[layer.fingerprint] = [layer]
        else:
            tasks[layer.fingerprint].append(layer)
    return tasks
