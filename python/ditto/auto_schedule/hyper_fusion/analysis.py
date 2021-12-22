import tvm
from typing import *
from ditto import hardware as hw
from ditto import utils
from .. import _ffi_api


BYTES_OF_TYPES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8
}


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


class AnalyticalResult(object):
    def __init__(self, locality, parallelism, recompute, valid):
        self.locality = locality
        self.parallelism = parallelism
        self.recompute = recompute
        self.valid = valid


def calculate_metrics(first_op_types: Tuple[str, str, str],
                      second_op_types: Tuple[str, str, str],
                      common_factors: List[Tuple[str, int]],
                      first_op_factors: List[Tuple[str, int]],
                      second_op_factors: List[Tuple[str, int]],
                      first_op_read_data_size: List[List[int]],
                      first_op_write_data_size: int,
                      second_op_read_data_size: List[List[int]],
                      second_op_write_data_size: int,
                      first_op_second_op_relate_pos: int,
                      redundant_common_factors: List[int],
                      hw_param: hw.HardwareParam):
    """Function to calculate the metrics locality, parallelism, and recompute.

    Args:
        first_op_types (Tuple[str, str, str]): input, output, compute types
        second_op_types (Tuple[str, str, str]): input, output, compute types
        common_factors (List[Tuple[str, int]]): e.g., [("S", 2), ("R", 4)]
        first_op_factors (List[Tuple[str, int]]):
        second_op_factors (List[Tuple[str, int]]):
        first_op_read_data_size (List[List[int]]):
        first_op_write_data_size (int):
        second_op_read_data_size (List[List[int]]):
        second_op_write_data_size (int):
        first_op_second_op_relate_pos (int):
        hw_param (hw.HardwareParam):
    """
    valid = True
    first_inp_byte = BYTES_OF_TYPES[first_op_types[0]]
    first_out_byte = BYTES_OF_TYPES[first_op_types[1]]
    second_inp_byte = BYTES_OF_TYPES[second_op_types[0]]
    second_out_byte = BYTES_OF_TYPES[second_op_types[1]]
    first_compute_weight = hw_param.get_compute_coeff(first_op_types[2])
    second_compute_weight = hw_param.get_compute_coeff(second_op_types[2])
    # the estimated shared memory usage for inputs
    # has already removed the effect of outer private reduce loops
    first_op_read_shared_memory_usage = utils.accumulate(
        [utils.accumulate(x) for x in first_op_read_data_size]
    )
    first_op_write_shared_memory_usage = first_op_write_data_size[0]
    second_op_read_shared_memory_usage = utils.accumulate(
        [utils.accumulate(x) for x in second_op_read_data_size]
    )
    # there is no need to cache the data for final outputs
    second_op_write_shared_memory_usage = 0
    assert first_op_second_op_relate_pos < len(second_op_read_data_size)
    # because of the fusion, there is no need to re-load the input for
    # the second op if it has already been produced from the first op
    second_op_read_shared_memory_usage -= utils.accumulate(
        second_op_read_data_size[first_op_second_op_relate_pos])

    total_shared_memory_usage = (
        first_op_read_shared_memory_usage * first_inp_byte +
        first_op_write_shared_memory_usage * first_out_byte +
        second_op_read_shared_memory_usage * second_inp_byte +
        second_op_write_shared_memory_usage * second_out_byte
    )

    valid = valid and (total_shared_memory_usage <
                       hw_param.shared_memory_per_block())

    # reduce factors in common iters can't be ignored because
    # we always do reduction within one block (not parallel)
    first_op_compute = utils.product(
        [1 if x[0] == "S" else x[1] for x in common_factors] + [x[1]
                                                                for x in first_op_factors]
    )
    second_op_compute = utils.product(
        [1 if x[0] == "S" else x[1] for x in common_factors] + [x[1]
                                                                for x in second_op_factors]
    )

    locality = ((first_op_compute * first_compute_weight + second_op_compute * second_compute_weight) /
                (first_op_read_shared_memory_usage * first_inp_byte + second_op_read_shared_memory_usage + second_inp_byte))
    sw_parallelism = utils.product(
        [1 if x[0] == "R" else x[1] for x in common_factors]
    )
    parallelism = min(
        sw_parallelism,
        # occupancy
        (hw_param.shared_memory_per_block() //
         total_shared_memory_usage) * hw_param.num_blocks()
    )
    recompute = utils.product(redundant_common_factors) * \
        first_op_compute * first_compute_weight

    return AnalyticalResult(
        locality,
        parallelism,
        recompute,
        valid
    )
