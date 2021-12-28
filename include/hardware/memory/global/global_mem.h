#pragma once

#include <hardware/memory/hw_memory.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for global memory.
 */
class GlobalMemoryNode : public HardwareMemoryNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.GlobalMemory";
  TVM_DECLARE_BASE_OBJECT_INFO(GlobalMemoryNode, HardwareMemoryNode);
}; // namespace hardware

class GlobalMemory : public HardwareMemory {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param capacity The size of this memory in kilo-bytes
   * \param pattern_list Allowed access patterns
   */
  TVM_DLL GlobalMemory(String name, double kb, Array<Pattern> pattern_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GlobalMemory, HardwareMemory,
                                        GlobalMemoryNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalMemoryNode);
};

} // namespace hardware

} // namespace ditto