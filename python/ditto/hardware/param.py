


class HardwareParam(object):
    """
    bandwidth: GB/s
    f32_peak_perf: GFLOPs
    launch_latency: (s)
    """
    def __init__(self, dram_bandwidth=None, f32_peak_perf=None, launch_latency=None):
        self.dram_bandwidth = dram_bandwidth
        self.f32_peak_perf = f32_peak_perf
        self.launch_latency = launch_latency