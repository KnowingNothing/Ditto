


class HardwareParam(object):
    """
    bandwidth: GB/s
    f32_peak_perf: GFLOPs
    launch_latency: (s)
    """
    def __init__(self, dram_bandwidth=None, f32_peak_perf=None, launch_latency=None, compute_coeff_dict=None,
                 shared_memory_per_block_byte=None, blocks=None):
        self.dram_bandwidth = dram_bandwidth
        self.f32_peak_perf = f32_peak_perf
        self.launch_latency = launch_latency
        self.compute_coeff_dict = compute_coeff_dict
        self.shared_memory_per_block_byte = shared_memory_per_block_byte
        self.blocks = blocks
        
    def get_compute_coeff(self, comp_type):
        return self.compute_coeff_dict[comp_type]
    
    def shared_memory_per_block(self):
        return self.shared_memory_per_block_byte
    
    def num_blocks(self):
        return self.blocks