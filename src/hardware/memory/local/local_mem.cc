#include <hardware/memory/local/local_mem.h>

namespace ditto {

namespace hardware {

LocalMemory::LocalMemory(String name, double kb, Array<te::Tensor> pattern) {
  auto node = make_object<LocalMemoryNode>();
  node->name = name;
  node->kb = kb;
  node->pattern = pattern;
  data_ = node;
}

} // namespace hardware

} // namespace ditto