"""
The fusion state implementation of hyper fusion
"""
import tvm
from . import _ffi_api
from typing import *
from .pattern import *
from .analysis import share_axis_analysis, calculate_metrics, AnalyticalResult
from .iter_graph import IV_TYPE_SPATIAL, IV_TYPE_REDUCE, IterVar, IterGraph, AccessFunc
from ditto import auto_compute as ac
from ditto import hardware as hw
from ditto import utils


class OpHyperState(object):
    """The state object for one op in hyper fusion.
    """

    def __init__(self, op, pattern: str):
        """
        Args:
            op (tvm.tensor.Operation): the operation
            pattern (str): PATTERN_XXX
        """
        self.op = op
        assert isinstance(self.op, tvm.te.tensor.ComputeOp)
        self.pattern = pattern

    def get_all_iters(self):
        """Get all IterVars for the op"""
        iters = []
        for i, iv in enumerate(self.op.axis):
            iters.append(IterVar(f"op({self.op.name},{hash(self.op)}).S{i}", i, ext=int(
                iv.dom.extent), iv_type=IV_TYPE_SPATIAL))
        for i, iv in enumerate(self.op.reduce_axis):
            iters.append(IterVar(f"op({self.op.name},{hash(self.op)}).R{i}", i, ext=int(
                iv.dom.extent), iv_type=IV_TYPE_REDUCE))
        return iters

    def get_all_iters_dict(self):
        iters_dict = {}
        for i, iv in enumerate(self.op.axis):
            iters_dict[iv] = (IterVar(f"op({self.op.name},{hash(self.op)}).S{i}", i, ext=int(
                iv.dom.extent), iv_type=IV_TYPE_SPATIAL))
        for i, iv in enumerate(self.op.reduce_axis):
            iters_dict[iv] = (IterVar(f"op({self.op.name},{hash(self.op)}).R{i}", i, ext=int(
                iv.dom.extent), iv_type=IV_TYPE_REDUCE))
        return iters_dict

    def get_read_access_functions(self):
        ret = []
        iters_dict = {iv.var: v for (
            iv, v) in self.get_all_iters_dict().items()}
        for inp in self.op.input_tensors:
            access_indices = utils.get_access_indices(self.op, inp)
            ret.append(AccessFunc(access_indices, iters_dict))
        return ret

    def get_write_access_functions(self):
        indices = [[iv.var for iv in self.op.axis]]
        iters_dict = {iv.var: v for (
            iv, v) in self.get_all_iters_dict().items()}
        return AccessFunc(indices, iters_dict)

    def get_first_producer_position(self):
        for i, inp in enumerate(self.op.input_tensors):
            if isinstance(inp.op, tvm.te.tensor.ComputeOp):
                return i
        return -1


class SerialHyperState(object):
    """The state object for hyper fusion.
        This state only models a linear topology.
    """

    def __init__(self, layer: ac.Layer):
        """
        Args:
            layer (ditto.auto_compute.Layer): the layer to fuse
        """
        self.layer = layer
        ops = []
        for op in layer.all_ops:
            if isinstance(op, tvm.te.tensor.ComputeOp):
                ops.append(op)
        self.ops = ops  # all the ops to fuse
        # the state of these ops
        # the values are OpHyperState
        self.op_states = [
            OpHyperState(op, get_op_pattern(op)) for op in self.ops
        ]
        self.validate()

    def validate(self):
        """Validate whether the layer can be fused."""
        # Check the validity
        feasible = True
        reason = ""
        # some logic to judge if the layer is suitable
        # 1. modify feasible if not suitable
        # 2. update the reason
        if not self.is_linear_topo():
            feasible = False
            reason = f"The given layer is not in linear topology.\nlayer info: {self.layer}."

        if self.count_cubic() != 2:
            feasible = False
            reason = f"Expect only 2 cubic operators in the layer, but get {self.count_cubic()}.\nlayer info: {self.layer}."

        if self.has_allred():
            feasible = False
            reason = ("Hyper fusion can't handle all reduce.\n The all reduce ops"
                      "should be eliminated by the frontend.\n Did you forget to use"
                      "Ditto's graph frontend?")

        if not feasible:
            raise ValueError(
                f"The given layer {self.layer} is not suitable for hyper fusion.\n{reason}.")

    def is_linear_topo(self):
        """Judge if a list of ops is in liner topology.

        Returns:
            [bool]: whether is in linear topology.
        """
        for op in self.ops:
            num_inputs = len(op.input_tensors)
            for i in range(num_inputs):
                for j in range(i + 1, num_inputs):
                    if (isinstance(op.input_tensors[i].op,
                                   tvm.te.tensor.ComputeOp) and
                        isinstance(op.input_tensors[j].op,
                                   tvm.te.tensor.ComputeOp) and
                            op.input_tensors[i].op != op.input_tensors[j].op):
                        return False
        return True

    def count_cubic(self):
        """Get the number of cubic ops in the critical path.

        Returns:
            [int]
        """
        counter = 0
        for op_state in self.op_states:
            if op_state.pattern == ac.nn.pattern.PATTERN_CUBIC:
                counter += 1
        return counter

    def has_allred(self):
        """Check if any op is allred pattern.

        Returns:
            [bool]
        """
        return any([state.pattern == ac.nn.pattern.PATTERN_ALLRED for state in self.op_states])

    def get_cubic_op_ids(self):
        ret = []
        for i, state in enumerate(self.op_states):
            if state.pattern == ac.nn.pattern.PATTERN_CUBIC:
                ret.append(i)
        return ret

    def build_iter_graph(self):
        """Build the IterGraph object."""
        # 0, 2
        first_op_id, second_op_id = self.get_cubic_op_ids()
        
        first_op_state = self.op_states[first_op_id]
        second_op_state = self.op_states[second_op_id]
        first_iters = first_op_state.get_all_iters()
        second_iters = second_op_state.get_all_iters()
        iters_dict = first_op_state.get_all_iters_dict()
        iters_dict.update(second_op_state.get_all_iters_dict())
        share_axis_pairs = share_axis_analysis(
            first_op_state.op, second_op_state.op)
        share_pairs = [
            (iters_dict[x[0]], iters_dict[x[1]]) for x in share_axis_pairs
        ]
        first_read_access = first_op_state.get_read_access_functions()
        first_write_access = first_op_state.get_write_access_functions()
        second_read_access = second_op_state.get_read_access_functions()
        second_write_access = second_op_state.get_write_access_functions()
        read_producer_pos = second_op_state.get_first_producer_position()
        assert read_producer_pos >= 0, "Can't find a producer for the second op."
        # build the iterator graph
        return IterGraph(first_iters,
                         second_iters,
                         share_pairs,
                         first_read_access,
                         second_read_access,
                         first_write_access,
                         second_write_access,
                         read_producer_pos)


def build_hyper_state(layer: ac.Layer):
    """This function validate whether the argument
        layer can be fused and auto-scheduled by the
        auto_schedule function of hyper_fusion.
        It reports error if the layer is not suitable.
        If suitable, it returns the hyper state of the layer.

        For now, we expect the layer to contain a list
        of ops in a linear topology:
        -->op1-->op2-->op3-->...-->opN-->
        And at most two of these ops are with cubic pattern.
    Args:
    ---
    layer (ditto.auto_compute.Layer): the layer to be fused and auto-scheduled

    Returns: hyper_state (SerialHyperState)
    ---
    """
    # build the initial hyper state
    hyper_state = SerialHyperState(layer)
    # update the hyper state according to some logic
    return hyper_state


def evaluate_iter_graph(iter_graph: IterGraph,
                        hw_param: hw.HardwareParam):
    """Evaluate the quality of an iter_graph

    Args:
        iter_graph (IterGraph): The iter graph
        hw_param (hw.HardwareParam): The target hardware

    Returns:
        [AnalyticalResult]: results
    """
    bounds = iter_graph.inferBound()
    common_iters = iter_graph.commonLoops()
    first_iters = iter_graph.firstLoops()
    second_iters = iter_graph.secondLoops()
    common_factors = [
        ("S" if iv.isSpatial() else "R", bounds[iv]) for iv in common_iters
    ]
    first_factors = [
        ("S" if iv.isSpatial() else "R", bounds[iv]) for iv in first_iters
    ]
    second_factors = [
        ("S" if iv.isSpatial() else "R", bounds[iv]) for iv in second_iters
    ]
    first_read = iter_graph.getFirstOpReadAccessDataSize(bounds)
    second_read = iter_graph.getSecondOpReadAccessDataSize(bounds)
    first_write = iter_graph.getFirstOpWriteAccessDataSize(bounds)
    second_write = iter_graph.getSecondOpWriteAccessDataSize(bounds)
    relate_pos = iter_graph.getFirstOpSecondOpRelateInputPos()
    redundant_factors = [bounds[iv]
                         for iv in iter_graph.redundantCommonLoops()]
    metric = calculate_metrics(
        # first_op_types: Tuple[str, str, str],
        ("float16", "float32", "float16-float32"),
        # second_op_types: Tuple[str, str, str],
        ("float16", "float32", "float32"),
        common_factors,  # common_factors: List[Tuple[str, int]],
        first_factors,  # first_op_factors: List[Tuple[str, int]],
        second_factors,  # second_op_factors: List[Tuple[str, int]],
        first_read,  # first_op_read_data_size: List[List[int]],
        first_write,  # first_op_write_data_size: int,
        second_read,  # second_op_read_data_size: List[List[int]],
        second_write,  # second_op_write_data_size: int,
        relate_pos,  # first_op_second_op_relate_pos: int,
        redundant_factors,  # redundant_common_factors: List[int],
        hw_param,  # hw_param: hw.HardwareParam
    )
    return metric

tvm._ffi._init_api("state", __name__)