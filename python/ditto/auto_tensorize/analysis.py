import tvm
from typing import *
from ditto import hardware as hw
from ditto import utils
from . import _ffi_api
from tvm.runtime import Object
from .iter_graph import IterGraph
from ditto.hardware.hw_param import HardwareParam


BYTES_OF_TYPES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}


def share_axis_analysis(
    op1: tvm.te.tensor.Operation,
    op2: tvm.te.tensor.Operation,
    tensorizeAxes: List[tvm.tir.IterVar] = [],
):
    """Perform the share relationship analysis.

    Args:
        op1 (tvm.te.tensor.Operation): the first op
        op2 (tvm.te.tensor.Operation): the second op

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
        List[List[tvm.tir.IterVar]]: share_axis_pairs
    """
    share_axis_pairs = _ffi_api.ShareAxisAnalysis(op1, op2, tensorizeAxes)
    return share_axis_pairs

@tvm._ffi.register_object("ditto.auto_tensorize.FusionResult")
class FusionResult(Object):
    def getLog(self):
        return _ffi_api.getLog(self)
