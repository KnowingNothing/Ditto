from .param import HardwareParam


V100 = HardwareParam(
    dram_bandwidth=750, # GiB/s
    f32_peak_perf=14*1e3, # GFLOPs
    launch_latency=5*1e-6, #s
    compute_coeff_dict={
            "float32": 1.0,
            "float16": 0.5,
            "float64": 2.0,
            "float16-float32": 1.5
        },
    shared_memory_per_block_byte=128*1e3,
    blocks=80
)