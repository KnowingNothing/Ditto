#include <hardware/base/hw_path.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwarePathNode);
TVM_REGISTER_NODE_TYPE(ComputePathNode);
TVM_REGISTER_NODE_TYPE(DataPathNode);

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

TVM_REGISTER_GLOBAL("ditto.hardware.ComputePath")
    .set_body_typed([](ISA isa, Pattern pattern, ISA load, ISA store) {
      return ComputePath(isa, pattern, load, store);
    });

TVM_REGISTER_GLOBAL("ditto.hardware.DataPath")
    .set_body_typed([](ISA isa, Pattern src_pattern, Pattern dst_pattern) {
      return DataPath(isa, src_pattern, dst_pattern);
    });

} // namespace hardware

} // namespace ditto