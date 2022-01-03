#include <hardware/base/hw_param.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareParamNode);

HardwareParam::HardwareParam(double dram_bandwidth_gbs,
                             double fp32_peak_perf_gflops,
                             double launch_latency_s,
                             double shared_memory_per_group_kb, int num_groups,
                             int num_processors_per_group) {
  auto node = make_object<HardwareParamNode>();
  node->dram_bandwidth_gbs = dram_bandwidth_gbs;
  node->fp32_peak_perf_gflops = fp32_peak_perf_gflops;
  node->launch_latency_s = launch_latency_s;
  node->shared_memory_per_group_kb = shared_memory_per_group_kb;
  node->num_groups = num_groups;
  node->num_processors_per_group = num_processors_per_group;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareParam")
    .set_body_typed([](double dram_bandwidth_gbs, double fp32_peak_perf_gflops,
                       double launch_latency_s,
                       double shared_memory_per_group_kb, int num_groups,
                       int num_processors_per_group) {
      return HardwareParam(dram_bandwidth_gbs, fp32_peak_perf_gflops,
                           launch_latency_s, shared_memory_per_group_kb,
                           num_groups, num_processors_per_group);
    });

} // namespace hardware

} // namespace ditto