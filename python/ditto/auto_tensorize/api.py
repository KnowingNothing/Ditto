"""
define the api of hyper fusion
"""

from ditto import hardware as hw
from ditto import auto_compute as ac
from .config import SUBSTANTIAL
from .space import FusionTileItem, FusionTileSpace
from .state import build_hyper_state


def auto_schedule(layer: ac.Layer, target: str):
    """This function performs automatic scheduling
        for the layer to be fused.

    Args:
    ---
    layer (ditto.auto_compute.Layer): the layer to be fused and auto-scheduled

    target (str): the string key to inform the auto_scheduler which hardware to schedule for.
        e.g., gpu.cuda.V100

    Returns: (new_layer, schedule, tensors)
    new_layer (ditto.auto_compute.Layer): the possibly modified new comptue
    schedule (tvm.te.Schedule): the schedule object of tvm for the new layer
    tensors (List[tvm.tensor.Tensor]): the tensors used to lower and build
    """
    hyper_state = build_hyper_state(layer)
    # hw_param = hw.query_hw_param(target)
    iter_graph = hyper_state.build_iter_graph()
    fuse_tile_space = FusionTileSpace(iter_graph, substantial=SUBSTANTIAL)
    # use the hw_param to schedule
    raise NotImplementedError()
