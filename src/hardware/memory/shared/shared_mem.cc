#include <hardware/memory/shared/shared_mem.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(SharedMemoryNode);

SharedMemory::SharedMemory(String name, double kb,
                           Map<String, Pattern> pattern_list) {
  auto node = make_object<SharedMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.SharedMemory")
    .set_body_typed([](String name, double kb,
                       Map<String, Pattern> pattern_list) {
      return SharedMemory(name, kb, pattern_list);
    });

} // namespace hardware

} // namespace ditto