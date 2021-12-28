#pragma once

#include <hardware/compute/unit/hw_unit.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A class for MMA unit.
 */
class MatrixMultiplyAccumulateNode : public HardwareUnitNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("isa_list", &isa_list);
  }

  static constexpr const char *_type_key =
      "ditto.hardware.MatrixMultiplyAccumulate";
  TVM_DECLARE_FINAL_OBJECT_INFO(MatrixMultiplyAccumulateNode, HardwareUnitNode);
};

class MatrixMultiplyAccumulate : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param isa_list The supported list of isa
   */
  TVM_DLL MatrixMultiplyAccumulate(String name, Array<ISA> isa_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MatrixMultiplyAccumulate, HardwareUnit,
                                        MatrixMultiplyAccumulateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatrixMultiplyAccumulateNode);
};

} // namespace hardware

} // namespace ditto