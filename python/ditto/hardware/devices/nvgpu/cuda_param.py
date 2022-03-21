from ...hw_param import *


"""
Bandwidth data from
"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
https://arxiv.org/abs/1804.06826
"""
V100_16GB_param = hardware_param(
    register_per_processor_kb = 256 / 4,  # register_per_processor_kb: float,
    shared_memory_per_group_kb = 96,  # shared_memory_per_group_kb: float, up to 96KB
    shared_memory_bandwidth_gbs = 12080,  # shared_memory_bandwidth_gbs: float,
    global_memory_gb = 16,  # global_memory_gb: float,
    global_memory_bandwidth_gbs = 750,  # global_memory_bandwidth_gbs: float,
    num_processors_per_group = 4,  # num_processors_per_group: int,
    num_groups = 80,  # num_groups: int,
    fp32_peak_perf_gflops = 14 * 1e3,  # fp32_peak_perf_gflops: float,
    launch_latency_s = 5 * 1e-6,  # launch_latency_s: float
)

V100_32GB_param = hardware_param(
    register_per_processor_kb = 256 / 4,  # register_per_processor_kb: float,
    shared_memory_per_group_kb = 96,  # shared_memory_per_group_kb: float, up to 96KB
    shared_memory_bandwidth_gbs = 12080,  # shared_memory_bandwidth_gbs: float,
    global_memory_gb = 32,  # global_memory_gb: float,
    global_memory_bandwidth_gbs = 750,  # global_memory_bandwidth_gbs: float,
    num_processors_per_group = 4,  # num_processors_per_group: int,
    num_groups = 80,  # num_groups: int,
    fp32_peak_perf_gflops = 14 * 1e3,  # fp32_peak_perf_gflops: float,
    launch_latency_s = 5 * 1e-6,  # launch_latency_s: float
)

"""
Bandwidth data is not measured.
The current data is copied from V100.
"""

A100_40GB_param = hardware_param(
    register_per_processor_kb = 256 / 4,  # register_per_processor_kb: float,
    shared_memory_per_group_kb = 164,  # shared_memory_per_group_kb: float, up to 164KB
    shared_memory_bandwidth_gbs = 12080,  # shared_memory_bandwidth_gbs: float,
    global_memory_gb = 40,  # global_memory_gb: float,
    global_memory_bandwidth_gbs = 1555,  # global_memory_bandwidth_gbs: float,
    num_processors_per_group = 4,  # num_processors_per_group: int,
    num_groups = 108,  # num_groups: int,
    fp32_peak_perf_gflops = 19.5 * 1e3,  # fp32_peak_perf_gflops: float,
    launch_latency_s = 5 * 1e-6,  # launch_latency_s: float
)

A100_80GB_param = hardware_param(
    register_per_processor_kb = 256 / 4,  # register_per_processor_kb: float,
    shared_memory_per_group_kb = 164,  # shared_memory_per_group_kb: float, up to 164KB
    shared_memory_bandwidth_gbs = 12080,  # shared_memory_bandwidth_gbs: float,
    global_memory_gb = 80,  # global_memory_gb: float,
    global_memory_bandwidth_gbs = 1555,  # global_memory_bandwidth_gbs: float,
    num_processors_per_group = 4,  # num_processors_per_group: int,
    num_groups = 108,  # num_groups: int,
    fp32_peak_perf_gflops = 19.5 * 1e3,  # fp32_peak_perf_gflops: float,
    launch_latency_s = 5 * 1e-6,  # launch_latency_s: float
)

"""
Bandwidth data is not measured.
The current data is copied from V100.
Configurations copied from RTX3080.
"""

RTX3090_param = hardware_param(
    register_per_processor_kb = 256 / 4,  # register_per_processor_kb: float,
    shared_memory_per_group_kb = 128,  # shared_memory_per_group_kb: float, up to 164KB
    shared_memory_bandwidth_gbs = 12080,  # shared_memory_bandwidth_gbs: float,
    global_memory_gb = 24,  # global_memory_gb: float,
    global_memory_bandwidth_gbs = 760,  # global_memory_bandwidth_gbs: float,
    num_processors_per_group = 4,  # num_processors_per_group: int,
    num_groups = 68,  # num_groups: int,
    fp32_peak_perf_gflops = 29.8 * 1e3,  # fp32_peak_perf_gflops: float,
    launch_latency_s = 5 * 1e-6,  # launch_latency_s: float
)