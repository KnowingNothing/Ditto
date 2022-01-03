#include <hardware/compute/group/homo_group.h>

namespace ditto {

namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareGroupNode);
TVM_REGISTER_NODE_TYPE(HomoGroupNode);

HomoGroup::HomoGroup(String name, HardwareProcessor processor,
                     SharedMemory shared_mem, int block_x, int block_y,
                     int block_z) {
  auto node = make_object<HomoGroupNode>();
  node->name = name;
  node->processor = processor;
  node->shared_mem = shared_mem;
  node->block_x = block_x;
  node->block_y = block_y;
  node->block_z = block_z;
  data_ = node;
}

TVM_REGISTER_GLOBAL("ditto.hardware.HomoGroup")
    .set_body_typed([](String name, HardwareProcessor processor,
                       SharedMemory shared_mem, int block_x, int block_y,
                       int block_z) {
      return HomoGroup(name, processor, shared_mem, block_x, block_y, block_z);
    });

} // namespace hardware

} // namespace ditto