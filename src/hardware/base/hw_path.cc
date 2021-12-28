#include <hardware/base/hw_path.h>

namespace ditto {

namespace hardware {

ComputePath::ComputePath(ISA isa, Pattern pattern, ISA load, ISA store) {
  auto node = make_object<ComputePathNode>();
  node->isa = isa;
  node->pattern = pattern;
  node->load = load;
  node->store = store;
  data_ = node;
}

DataPath::DataPath(ISA isa, Pattern src_pattern, Pattern dst_pattern) {
  auto node = make_object<DataPathNode>();
  node->isa = isa;
  node->src_pattern = src_pattern;
  node->dst_pattern = dst_pattern;
  data_ = node;
}

} // namespace hardware

} // namespace ditto