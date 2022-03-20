#include <hardware/base/hw_param.h>

namespace ditto
{

  namespace hardware
  {

    TVM_REGISTER_NODE_TYPE(HardwareParamNode);

    HardwareParam::HardwareParam(
        double register_per_processor_kb,
        double shared_memory_per_group_kb,
        double shared_memory_bandwidth_gbs,
        double global_memory_gb,
        double global_memory_bandwidth_gbs,
        int num_processors_per_group, int num_groups,
        double fp32_peak_perf_gflops, double launch_latency_s,
        std::vector<double> tensorWeight,
        std::vector<double> cacheSizes,
        std::vector<double> bandwidth,
        std::vector<double> coresPerCacheLevel,
        std::string platform)
    {
      CHECK(coresPerCacheLevel.size() == cacheSizes.size());
      auto node = make_object<HardwareParamNode>();
      node->tensorWeight = tensorWeight;
      node->register_per_processor_kb = register_per_processor_kb;
      node->shared_memory_per_group_kb = shared_memory_per_group_kb;
      node->shared_memory_bandwidth_gbs = shared_memory_bandwidth_gbs;
      node->global_memory_gb = global_memory_gb;
      node->global_memory_bandwidth_gbs = global_memory_bandwidth_gbs;
      node->num_processors_per_group = num_processors_per_group;
      node->num_groups = num_groups;
      node->fp32_peak_perf_gflops = fp32_peak_perf_gflops;
      node->launch_latency_s = launch_latency_s;
      node->cacheSizes = cacheSizes;
      node->cacheBandwidth = bandwidth;
      node->platform = platform;
      node->coresPerCacheLevel = coresPerCacheLevel;
      for (size_t i = 0; i < cacheSizes.size(); i++)
        node->cacheSizePerThread.push_back(cacheSizes[i] / coresPerCacheLevel[i]);
      data_ = node;
    }

    TVM_REGISTER_GLOBAL("ditto.hardware.HardwareParam")
        .set_body_typed(
            [](double register_per_processor_kb, double shared_memory_per_group_kb,
               double shared_memory_bandwidth_gbs, double global_memory_gb,
               double global_memory_bandwidth_gbs, int num_processors_per_group,
               int num_groups, double fp32_peak_perf_gflops,
               double launch_latency_s, Array<FloatImm> tensorWeight = {},
               Array<FloatImm> cacheSizes = {}, Array<FloatImm> bandwidth = {},
               Array<FloatImm> coresPerCacheLevel = {},
               String platform = "NVGPU")
            {
              std::vector<double> tensorWeight_;
              for (auto ts : tensorWeight)
                tensorWeight_.push_back(ts->value);
              std::vector<double> cacheSizes_;
              for (auto cs : cacheSizes)
                cacheSizes_.push_back(cs->value);
              std::vector<double> bandwidth_;
              for (auto bd : bandwidth)
                bandwidth_.push_back(bd->value);
              std::vector<double> coresPerCacheLevel_;
              for (auto _ : coresPerCacheLevel)
                coresPerCacheLevel_.push_back(_->value);
              return HardwareParam(
                  register_per_processor_kb, shared_memory_per_group_kb,
                  shared_memory_bandwidth_gbs, global_memory_gb,
                  global_memory_bandwidth_gbs, num_processors_per_group, num_groups,
                  fp32_peak_perf_gflops, launch_latency_s, tensorWeight_,
                  cacheSizes_, bandwidth_, coresPerCacheLevel_, std::string(platform));
            });

  } // namespace hardware

} // namespace ditto