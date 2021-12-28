#pragma once

#include <hardware/compute/hw_compute.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware processor.
 */
class HardwareProcessorNode : public HardwareComputeNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.HardwareProcessor";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareProcessorNode, HardwareComputeNode);
};

class HardwareProcessor : public HardwareCompute {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareProcessor, HardwareCompute,
                                        HardwareProcessorNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareProcessorNode);
};

} // namespace hardware

} // namespace ditto