#pragma once

#include <hardware/compute/hw_compute.h>
#include <hardware/compute/processor/hw_processor.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware group.
 */
class HardwareGroupNode : public HardwareComputeNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.HardwareGroup";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareGroupNode, HardwareNode);
};

class HardwareGroup : public HardwareCompute {
public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareGroup, HardwareCompute,
                                        HardwareGroupNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareGroupNode);
};

} // namespace hardware

} // namespace ditto