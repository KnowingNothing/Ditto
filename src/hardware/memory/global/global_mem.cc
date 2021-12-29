#include <hardware/memory/global/global_mem.h>

namespace ditto {

namespace hardware {

GlobalMemory::GlobalMemory(String name, double gb,
                           Map<String, Pattern> pattern_list) {
  auto node = make_object<GlobalMemoryNode>();
  node->name = name;
  node->kb = gb * 1e6;
  node->pattern_list = pattern_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto