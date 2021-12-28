#include <hardware/memory/hw_memory.h>

namespace ditto {

namespace hardware {

HardwareMemory::HardwareMemory(String name, double kb,
                               Array<Pattern> pattern_list) {
  auto node = make_object<HardwareMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto