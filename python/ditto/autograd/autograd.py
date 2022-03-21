from . import _ffi_api


def grad_layer(layer):
    """
    get the gradient of a layer

    Args:
    ---
    layer: ditto.auto_compute.Layer

    Returns:
    ---
    ditto.auto_compute.Layer
    """
    return _ffi_api.GradLayer(layer)


def grad_graph(graph, reserve_forward=False):
    """
    get the gradient of a graph

    Args:
    ---
    graph: ditto.auto_compute.Graph

    Returns:
    ---
    ditto.auto_compute.Graph
    """
    return _ffi_api.GradGraph(graph, reserve_forward)
