#pragma once

#include <hardware/base/hw_base.h>
#include <hardware/base/pattern_base.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware memory.
 */
class HardwareMemoryNode : public HardwareNode {
public:
  double kb;
  Map<String, Pattern> pattern_list;

  static constexpr const char *_type_key = "ditto.hardware.HardwareMemory";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareMemoryNode, HardwareNode);
};

class HardwareMemory : public Hardware {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param capacity The size of this memory in kilo-bytes
   * \param pattern_list The supported access patterns
   */
  TVM_DLL HardwareMemory(String name, double kb,
                         Map<String, Pattern> pattern_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareMemory, Hardware,
                                        HardwareMemoryNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareMemoryNode);
};

} // namespace hardware

} // namespace ditto