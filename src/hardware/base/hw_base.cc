#include <hardware/hw_model.h>

namespace ditto {

namespace hardware {

Hardware::Hardware(String name) {
  auto node = make_object<HardwareNode>();
  node->name = name;
  data_ = node;
}

} // namespace hardware

} // namespace ditto