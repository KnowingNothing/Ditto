#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A class for general-purpose register files.
 */
class FragmentNode : public LocalMemoryNode {
public:
  static constexpr const char *_type_key = "ditto.hardware.Fragment";
  TVM_DECLARE_BASE_OBJECT_INFO(FragmentNode, LocalMemoryNode);
}; // namespace hardware

class Fragment : public LocalMemory {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param capacity The size of this memory in kilo-bytes
   * \param pattern Allowed view of the memory
   */
  TVM_DLL Fragment(String name, double kb, Array<te::Tensor> pattern);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Fragment, LocalMemory, FragmentNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FragmentNode);
};

} // namespace hardware

} // namespace ditto