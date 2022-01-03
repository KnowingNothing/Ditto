import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.HardwareParam")
class HardwareParam(Object):
    """HardwareParam object"""


def hardware_param(dram_bandwidth_gbs: float,
                   fp32_peak_perf_gflops: float,
                   launch_latency_s: float,
                   shared_memory_per_group_kb: float,
                   num_groups: int,
                   num_processors_per_group: int):
    """Get hardware params

    Args:
        dram_bandwidth_gbs (float): GB/s
        fp32_peak_perf_gflops (float): GFLOPs
        launch_latency_s (float): second
        shared_memory_per_group_kb (float): KB
        num_groups (int):
        num_processors_per_group (int):

    Returns:
        hardware.HardwareParam
    """
    return _ffi_api.HardwareParam(
        dram_bandwidth_gbs,
        fp32_peak_perf_gflops,
        launch_latency_s,
        shared_memory_per_group_kb,
        num_groups,
        num_processors_per_group
    )
