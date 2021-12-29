#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

LocalMemory::LocalMemory(String name, double kb, Map<String, Pattern> pattern_list) {
  auto node = make_object<LocalMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto