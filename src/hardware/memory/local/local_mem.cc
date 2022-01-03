#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(LocalMemoryNode);

LocalMemory::LocalMemory(String name, double kb,
                         Map<String, Pattern> pattern_list) {
  auto node = make_object<LocalMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.LocalMemory")
    .set_body_typed([](String name, double kb,
                       Map<String, Pattern> pattern_list) {
      return LocalMemory(name, kb, pattern_list);
    });

} // namespace hardware

} // namespace ditto