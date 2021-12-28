#include <hardware/memory/shared/shared_mem.h>

namespace ditto {

namespace hardware {

SharedMemory::SharedMemory(String name, double kb,
                           Array<Pattern> pattern_list) {
  auto node = make_object<SharedMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern_list = pattern_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto