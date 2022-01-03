#include <hardware/compute/unit/hw_unit.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareUnitNode);

HardwareUnit::HardwareUnit(String name, Map<String, ISA> isa_list) {
  auto node = make_object<HardwareUnitNode>();
  node->name = name;
  node->isa_list = isa_list;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareUnit")
    .set_body_typed([](String name, Map<String, ISA> isa_list) {
      return HardwareUnit(name, isa_list);
    });

} // namespace hardware

} // namespace ditto