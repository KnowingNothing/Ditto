from .param import HardwareParam


V100 = HardwareParam(
    dram_bandwidth=750, # GiB/s
    f32_peak_perf=14*1e3, # GFLOPs
    launch_latency=5*1e-6, #s
)