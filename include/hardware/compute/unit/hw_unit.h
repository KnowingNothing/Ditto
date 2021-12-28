#pragma once

#include <hardware/base/visa_base.h>
#include <hardware/compute/hw_compute.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware unit.
 */
class HardwareUnitNode : public HardwareComputeNode {
public:
  Array<ISA> isa_list;

  static constexpr const char *_type_key = "ditto.hardware.HardwareUnit";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareUnitNode, HardwareComputeNode);
};

class HardwareUnit : public HardwareCompute {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param isa_list The list of supported isa
   */
  TVM_DLL HardwareUnit(String name, Array<ISA> isa_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareUnit, HardwareCompute,
                                        HardwareUnitNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareUnitNode);
};

} // namespace hardware

} // namespace ditto