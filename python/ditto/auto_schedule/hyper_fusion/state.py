"""
The fusion state implementation of hyper fusion
"""
import tvm
from .pattern import *
from .iter_graph import IV_TYPE_SPATIAL, IV_TYPE_REDUCE, IterVar, IterGraph
from ditto import auto_compute as ac


def share_axis_analysis(op1, op2):
    """Perform the share relationship analysis.

    Args:
        op1 (tvm.tensor.Operation): the first op
        op2 (tvm.tensor.Operation): the second op

        op1 and op2 may not have direct access relationship,
        i.e., the possible topology is
        op1-->op?-->op?-->...-->op2.
        So in this analysis, the share relationship should
        be propagated along the producer-consumer chain.

    Example:
        GEMM1 C[i, l] = A[i, k] * B[k ,l]
        ReLU  D[i', l'] = ReLU(C[i', l'])
        GEMM2 F[i'', j] = D[i, l''] * E[l'', j]

        op1 is GEMM1, op2 is GEMM2

        The result is [(i, i''), (l, l'')]

    Returns:
        List[Tuple(IterVar, IterVar)]
    """
    # Note: the IterVar in result shared_paris should use
    # the same naming format as OpHyperState
    # 'Si' for the i-th spatial axis, 'Ri' for the i-th reduce axis
    raise NotImplementedError()


class OpHyperState(object):
    """The state object for one op in hyper fusion.
    """

    def __init__(self, op, pattern):
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
            iters.append(IterVar(f"S{i}", ext=int(
                iv.dom.ext), iv_type=IV_TYPE_SPATIAL))
        for i, iv in enumerate(self.op.reduce_axis):
            iters.append(IterVar(f"R{i}", ext=int(
                iv.dom.ext), iv_type=IV_TYPE_REDUCE))
        return iters


class SerialHyperState(object):
    """The state object for hyper fusion.
        This state only models a linear topology.
    """

    def __init__(self, layer):
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
        self.iter_graph = self.build_iter_graph()

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
            reason = "The given layer is not in linear topology.\nlayer info: {self.layer}."

        if self.count_cubic() == 2:
            feasible = False
            reason = "Expect only 2 cubic operators in the layer.\nlayer info: {self.layer}."

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
        raise NotImplementedError()

    def count_cubic(self):
        """Get the number of cubic ops in the critical path.

        Returns:
            [int]
        """
        raise NotImplementedError()

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
        first_op_id, second_op_id = self.get_cubic_op_ids()
        first_op_state = self.op_states[first_op_id]
        second_op_state = self.op_states[second_op_id]
        first_iters = first_op_state.get_all_iters()
        second_iters = second_op_state.get_all_iters()
        share_pairs = share_axis_analysis(
            first_op_state.op, second_op_state.op)
        # build the iterator graph
        self.iter_graph = IterGraph(first_iters, second_iters, share_pairs)


def build_hyper_state(layer):
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
