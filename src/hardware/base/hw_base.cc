#include <hardware/base/hw_base.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareNode);

Hardware::Hardware(String name) {
  auto node = make_object<HardwareNode>();
  node->name = name;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.Hardware").set_body_typed([](String name) {
  return Hardware(name);
});

} // namespace hardware

} // namespace ditto