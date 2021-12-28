#pragma once

#include <hardware/compute/group/hw_group.h>
#include <hardware/compute/hw_compute.h>
#include <hardware/memory/global/global_mem.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware device.
 */
class HardwareDeviceNode : public HardwareComputeNode {
public:
  /*! \brief The SM */
  HardwareGroup group;
  /*! \brief The global memory */
  GlobalMemory global_mem;
  /*! \brief The x dim size */
  int grid_x;
  /*! \brief The y dim size */
  int grid_y;
  /*! \brief The z dim size */
  int grid_z;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("sm", &group);
    v->Visit("device_mem", &global_mem);
    v->Visit("grid_x", &grid_x);
    v->Visit("grid_y", &grid_y);
    v->Visit("block_z", &grid_z);
  }
  static constexpr const char *_type_key = "ditto.hardware.HardwareDevice";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareDeviceNode, HardwareNode);
};

class HardwareDevice : public HardwareCompute {
public:
  TVM_DLL HardwareDevice(String name, HardwareGroup group,
                         GlobalMemory global_mem, int grid_x, int grid_y,
                         int grid_z);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareDevice, HardwareCompute,
                                        HardwareDeviceNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareDeviceNode);
};

} // namespace hardware

} // namespace ditto