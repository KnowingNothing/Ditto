#pragma once

#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware parameters.
 */
class HardwareParamNode : public Object {
public:
  double register_per_processor_kb;
  double shared_memory_per_group_kb;
  double shared_memory_bandwidth_gbs;
  double global_memory_gb;
  double global_memory_bandwidth_gbs;
  int num_processors_per_group;
  int num_groups;
  double fp32_peak_perf_gflops;
  double launch_latency_s;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("register_per_processor_kb", &register_per_processor_kb);
    v->Visit("shared_memory_per_group_kb", &shared_memory_per_group_kb);
    v->Visit("shared_memory_bandwidth_gbs", &shared_memory_bandwidth_gbs);
    v->Visit("global_memory_gb", &global_memory_gb);
    v->Visit("dram_bandwidth_gbs", &global_memory_bandwidth_gbs);
    v->Visit("num_processors_per_group", &num_processors_per_group);
    v->Visit("num_groups", &num_groups);
    v->Visit("fp32_peak_perf_gflops", &fp32_peak_perf_gflops);
    v->Visit("launch_latency_s", &launch_latency_s);
  }

  static constexpr const char *_type_key = "ditto.hardware.HardwareParam";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareParamNode, Object);
};

class HardwareParam : public ObjectRef {
public:
  TVM_DLL HardwareParam(double register_per_processor_kb,
                        double shared_memory_per_group_kb,
                        double shared_memory_bandwidth_gbs,
                        double global_memory_gb,
                        double global_memory_bandwidth_gbs,
                        int num_processors_per_group, int num_groups,
                        double fp32_peak_perf_gflops, double launch_latency_s);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareParam, ObjectRef,
                                        HardwareParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareParamNode);
};

} // namespace hardware

} // namespace ditto