#include <hardware/memory/hw_memory.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareMemoryNode);

HardwareMemory::HardwareMemory(String name, double kb,
                               Map<String, Pattern> pattern_list) {
  auto node = make_object<HardwareMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareMemory")
    .set_body_typed([](String name, double kb,
                       Map<String, Pattern> pattern_list) {
      return HardwareMemory(name, kb, pattern_list);
    });

} // namespace hardware

} // namespace ditto