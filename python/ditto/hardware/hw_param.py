import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.HardwareParam")
class HardwareParam(Object):
    """HardwareParam object"""


def hardware_param(register_per_processor_kb: float,
                   shared_memory_per_group_kb: float,
                   shared_memory_bandwidth_gbs: float,
                   global_memory_gb: float,
                   global_memory_bandwidth_gbs: float,
                   num_processors_per_group: int,
                   num_groups: int,
                   fp32_peak_perf_gflops: float,
                   launch_latency_s: float):
    """The hardware params

    Args:
        register_per_processor_kb (float): KB
        shared_memory_per_group_kb (float): KB
        shared_memory_bandwidth_gbs (float): GB/s
        global_memory_gb (float): GB
        global_memory_bandwidth_gbs (float): GB
        num_processors_per_group (int):
        num_groups (int):
        fp32_peak_perf_gflops (float): GFLOPs
        launch_latency_s (float): second

    Returns:
        HardwareParam
    """
    return _ffi_api.HardwareParam(
        register_per_processor_kb, shared_memory_per_group_kb,
        shared_memory_bandwidth_gbs, global_memory_gb,
        global_memory_bandwidth_gbs, num_processors_per_group, num_groups,
        fp32_peak_perf_gflops, launch_latency_s
    )
