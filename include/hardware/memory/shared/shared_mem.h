#pragma once

#include <hardware/memory/hw_memory.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for shared memory.
 */
class SharedMemoryNode : public HardwareMemoryNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("kb", &kb);
    v->Visit("pattern_list", &pattern_list);
  }

  static constexpr const char *_type_key = "ditto.hardware.SharedMemory";
  TVM_DECLARE_BASE_OBJECT_INFO(SharedMemoryNode, HardwareMemoryNode);
}; // namespace hardware

class SharedMemory : public HardwareMemory {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param capacity The size of this memory in kilo-bytes
   * \param pattern_list Allowed access patterns
   */
  TVM_DLL SharedMemory(String name, double kb,
                       Map<String, Pattern> pattern_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SharedMemory, HardwareMemory,
                                        SharedMemoryNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SharedMemoryNode);
};

} // namespace hardware

} // namespace ditto