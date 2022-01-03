#include <hardware/compute/device/hw_device.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareDeviceNode);

HardwareDevice::HardwareDevice(String name, HardwareGroup group,
                               GlobalMemory global_mem, int grid_x, int grid_y,
                               int grid_z) {
  auto node = make_object<HardwareDeviceNode>();
  node->name = name;
  node->group = group;
  node->global_mem = global_mem;
  node->grid_x = grid_x;
  node->grid_y = grid_y;
  node->grid_z = grid_z;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HardwareDevice")
    .set_body_typed([](String name, HardwareGroup group,
                       GlobalMemory global_mem, int grid_x, int grid_y,
                       int grid_z) {
      return HardwareDevice(name, group, global_mem, grid_x, grid_y, grid_z);
    });

} // namespace hardware

} // namespace ditto