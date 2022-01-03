from .hw_param import hardware_param


V100 = hardware_param(
    750,  # dram_bandwidth_gbs,
    14*1e3,  # fp32_peak_perf_gflops,
    5*1e-6,  # launch_latency_s,
    128,  # shared_memory_per_group_kb,
    80,  # num_groups,
    4,  # num_processors_per_group
)
