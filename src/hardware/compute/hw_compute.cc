#include <hardware/compute/hw_compute.h>

namespace ditto {

namespace hardware {

HardwareCompute::HardwareCompute(String name) {
  auto node = make_object<HardwareComputeNode>();
  node->name = name;
  data_ = node;
}

} // namespace hardware

} // namespace ditto