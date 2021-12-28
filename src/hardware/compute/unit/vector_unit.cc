#include <hardware/compute/unit/vector_unit.h>

namespace ditto {

namespace hardware {

VectorAdder::VectorAdder(String name, Array<ISA> isa_list) {
  auto node = make_object<VectorAdderNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto