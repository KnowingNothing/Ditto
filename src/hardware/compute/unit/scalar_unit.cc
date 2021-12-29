#include <hardware/compute/unit/scalar_unit.h>

namespace ditto {

namespace hardware {

Adder::Adder(String name, Map<String, ISA> isa_list) {
  auto node = make_object<AdderNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

Multiplier::Multiplier(String name, Map<String, ISA> isa_list) {
  auto node = make_object<MultiplierNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

MultiplyAdder::MultiplyAdder(String name, Map<String, ISA> isa_list) {
  auto node = make_object<MultiplyAdderNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto