#include <hardware/compute/device/hw_device.h>

namespace ditto {

namespace hardware {

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

} // namespace hardware

} // namespace ditto