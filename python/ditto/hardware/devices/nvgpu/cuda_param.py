from ...hw_param import *


"""
bandwidth data from
"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
https://arxiv.org/abs/1804.06826
"""
V100_16GB_param = hardware_param(
    256 / 4,  # register_per_processor_kb: float,
    96,  # shared_memory_per_group_kb: float, up to 96KB
    12080,  # shared_memory_bandwidth_gbs: float,
    16,  # global_memory_gb: float,
    750,  # global_memory_bandwidth_gbs: float,
    4,  # num_processors_per_group: int,
    80,  # num_groups: int,
    14 * 1e3,  # fp32_peak_perf_gflops: float,
    5 * 1e-6,  # launch_latency_s: float
)

V100_32GB_param = hardware_param(
    256 / 4,  # register_per_processor_kb: float,
    96,  # shared_memory_per_group_kb: float, up to 96KB
    12080,  # shared_memory_bandwidth_gbs: float,
    32,  # global_memory_gb: float,
    750,  # global_memory_bandwidth_gbs: float,
    4,  # num_processors_per_group: int,
    80,  # num_groups: int,
    14 * 1e3,  # fp32_peak_perf_gflops: float,
    5 * 1e-6,  # launch_latency_s: float
)
