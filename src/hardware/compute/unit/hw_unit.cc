#include <hardware/compute/unit/hw_unit.h>

namespace ditto {

namespace hardware {

HardwareUnit::HardwareUnit(String name, double latency,
                           Array<te::Operation> functionality) {
  auto node = make_object<HardwareUnitNode>();
  node->name = name;
  node->latency = latency;
  node->functionality = functionality;
  data_ = node;
}

} // namespace hardware

} // namespace ditto