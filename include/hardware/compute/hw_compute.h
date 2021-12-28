#pragma once

#include <hardware/base/hw_base.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware compute.
 */
class HardwareComputeNode : public HardwareNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.HardwareCompute";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareComputeNode, HardwareNode);
};

class HardwareCompute : public Hardware {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   */
  TVM_DLL HardwareCompute(String name);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareCompute, Hardware,
                                        HardwareComputeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareComputeNode);
};

} // namespace hardware

} // namespace ditto