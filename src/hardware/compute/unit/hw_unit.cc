#include <hardware/compute/unit/hw_unit.h>

namespace ditto {

namespace hardware {

HardwareUnit::HardwareUnit(String name, Array<ISA> isa_list) {
  auto node = make_object<HardwareUnitNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

} // namespace hardware

} // namespace ditto