import tvm
from typing import *
from .. import _ffi_api


def share_axis_analysis(
    op1: tvm.te.tensor.Operation,
    op2: tvm.te.tensor.Operation,
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
    share_axis_pairs = _ffi_api.ShareAxisAnalysis(op1, op2)
    return share_axis_pairs


def calculate_metrics(self,
                      common_factors: List[Tuple[str, int]],
                      first_op_factors: List[Tuple[str, int]],
                      second_op_factors: List[Tuple[str, int]],
                      first_op_access_data_size: List[List[int]],
                      second_op_access_data_size: List[List[int]]):
    pass
