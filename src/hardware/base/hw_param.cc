#include <hardware/base/hw_param.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareParamNode);

HardwareParam::HardwareParam(
    double register_per_processor_kb, double shared_memory_per_group_kb,
    double shared_memory_bandwidth_gbs, double global_memory_gb,
    double global_memory_bandwidth_gbs, int num_processors_per_group,
    int num_groups, double fp32_peak_perf_gflops, double launch_latency_s) {
  auto node = make_object<HardwareParamNode>();
  node->register_per_processor_kb = register_per_processor_kb;
  node->shared_memory_per_group_kb = shared_memory_per_group_kb;
  node->shared_memory_bandwidth_gbs = shared_memory_bandwidth_gbs;
  node->global_memory_gb = global_memory_gb;
  node->global_memory_bandwidth_gbs = global_memory_bandwidth_gbs;
  node->num_processors_per_group = num_processors_per_group;
  node->num_groups = num_groups;
  node->fp32_peak_perf_gflops = fp32_peak_perf_gflops;
  node->launch_latency_s = launch_latency_s;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareParam")
    .set_body_typed([](double register_per_processor_kb,
                       double shared_memory_per_group_kb,
                       double shared_memory_bandwidth_gbs,
                       double global_memory_gb,
                       double global_memory_bandwidth_gbs,
                       int num_processors_per_group, int num_groups,
                       double fp32_peak_perf_gflops, double launch_latency_s) {
      return HardwareParam(
          register_per_processor_kb, shared_memory_per_group_kb,
          shared_memory_bandwidth_gbs, global_memory_gb,
          global_memory_bandwidth_gbs, num_processors_per_group, num_groups,
          fp32_peak_perf_gflops, launch_latency_s);
    });

} // namespace hardware

} // namespace ditto