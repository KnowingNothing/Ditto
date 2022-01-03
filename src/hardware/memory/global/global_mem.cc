#include <hardware/memory/global/global_mem.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(GlobalMemoryNode);

GlobalMemory::GlobalMemory(String name, double gb,
                           Map<String, Pattern> pattern_list) {
  auto node = make_object<GlobalMemoryNode>();
  node->name = name;
  node->kb = gb * 1e6;
  node->pattern_list = pattern_list;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.GlobalMemory")
    .set_body_typed([](String name, double kb,
                       Map<String, Pattern> pattern_list) {
      return GlobalMemory(name, kb, pattern_list);
    });

} // namespace hardware

} // namespace ditto