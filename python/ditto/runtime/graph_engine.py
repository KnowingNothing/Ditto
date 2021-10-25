from . import _ffi_api


def create_graph_engine(graph, built_mods, dev):
    """
    Create graph engine

    Args:
    ---
    graph: ditto.auto_compute.Graph

    built_mods: Map{wkl_key, tvm.runtime.Module}

    dev: tvm.runtime.Device

    Return:
    ---
    ditto.runtime.GraphEngine
    """
    return _ffi_api.create_graph_engine(graph, built_mods, dev)
