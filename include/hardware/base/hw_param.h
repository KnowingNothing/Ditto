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
  // self.dram_bandwidth = dram_bandwidth
  //       self.f32_peak_perf = f32_peak_perf
  //       self.launch_latency = launch_latency
  //       self.compute_coeff_dict = compute_coeff_dict
  //       self.shared_memory_per_block_byte = shared_memory_per_block_byte
  //       self.blocks = blocks
  double dram_bandwidth_gbs;
  double fp32_peak_perf_gflops;
  double launch_latency_s;
  double shared_memory_per_group_kb;
  int num_groups;
  int num_processors_per_group;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("dram_bandwidth_gbs", &dram_bandwidth_gbs);
    v->Visit("fp32_peak_perf_gflops", &fp32_peak_perf_gflops);
    v->Visit("launch_latency_s", &launch_latency_s);
    v->Visit("shared_memory_per_group_kb", &shared_memory_per_group_kb);
    v->Visit("num_groups", &num_groups);
    v->Visit("num_processors_per_group", &num_processors_per_group);
  }

  static constexpr const char *_type_key = "ditto.hardware.HardwareParam";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareParamNode, Object);
};

class HardwareParam : public ObjectRef {
public:
  TVM_DLL HardwareParam(double dram_bandwidth_gbs, double fp32_peak_perf_gflops,
                        double launch_latency_s,
                        double shared_memory_per_group_kb, int num_groups,
                        int num_processors_per_group);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareParam, ObjectRef,
                                        HardwareParamNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareParamNode);
};

} // namespace hardware

} // namespace ditto