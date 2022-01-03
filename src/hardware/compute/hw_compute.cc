#include <hardware/compute/hw_compute.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareComputeNode);

HardwareCompute::HardwareCompute(String name) {
  auto node = make_object<HardwareComputeNode>();
  node->name = name;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareCompute")
    .set_body_typed([](String name) { return HardwareCompute(name); });

} // namespace hardware

} // namespace ditto