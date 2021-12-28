#pragma once

#include <hardware/compute/group/hw_group.h>
#include <hardware/compute/processor/hw_processor.h>
#include <hardware/memory/shared/shared_mem.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A class for homogeneous group.
 */
class HomoGroupNode : public HardwareGroupNode {
public:
  /*! \brief The hardware processor */
  HardwareProcessor processor;
  /*! \brief The shared memory */
  SharedMemory shared_mem;
  /*! \brief The x dim size */
  int block_x;
  /*! \brief The y dim size */
  int block_y;
  /*! \brief The z dim size */
  int block_z;

  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("processor", &processor);
    v->Visit("shared_mem", &shared_mem);
    v->Visit("block_x", &block_x);
    v->Visit("block_y", &block_y);
    v->Visit("block_z", &block_z);
  }

  static constexpr const char *_type_key = "ditto.hardware.HomoGroup";
  TVM_DECLARE_FINAL_OBJECT_INFO(HomoGroupNode, HardwareGroupNode);
};

class HomoGroup : public HardwareGroup {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param processor The hardware processor
   * \param shared_mem The shared memory
   * \param block_x The x dim size
   * \param block_y The y dim size
   * \param block_z The z dim size
   */
  TVM_DLL HomoGroup(String name, HardwareProcessor processor,
                    SharedMemory shared_mem, int block_x, int block_y,
                    int block_z);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HomoGroup, HardwareGroup,
                                        HomoGroupNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HomoGroupNode);
};

} // namespace hardware

} // namespace ditto