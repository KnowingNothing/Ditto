#pragma once

#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A class for general-purpose register files.
 */
class RegisterArrayNode : public LocalMemoryNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.RegisterArray";
  TVM_DECLARE_FINAL_OBJECT_INFO(RegisterArrayNode, LocalMemoryNode);
}; // namespace hardware

class RegisterArray : public LocalMemory {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param capacity The size of this memory in kilo-bytes
   * \param pattern_list Allowed access patterns
   */
  TVM_DLL RegisterArray(String name, double kb,
                        Map<String, Pattern> pattern_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RegisterArray, LocalMemory,
                                        RegisterArrayNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RegisterArrayNode);
};

} // namespace hardware

} // namespace ditto