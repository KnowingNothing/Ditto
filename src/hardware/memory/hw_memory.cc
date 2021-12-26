#include <hardware/memory/hw_memory.h>

namespace ditto {

namespace hardware {

HardwareMemory::HardwareMemory(String name, double kb) {
  auto node = make_object<HardwareMemoryNode>();
  node->name = name;
  node->kb = kb;
  data_ = node;
}

} // namespace hardware

} // namespace ditto