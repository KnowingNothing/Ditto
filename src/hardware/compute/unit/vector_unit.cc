#include <hardware/compute/unit/vector_unit.h>

namespace ditto {

namespace hardware {

VectorAdder::VectorAdder(String name, double latency,
                         Array<te::Operation> functionality) {
  auto node = make_object<VectorAdderNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

} // namespace hardware

} // namespace ditto