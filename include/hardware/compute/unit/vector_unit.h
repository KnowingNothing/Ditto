#pragma once

#include <hardware/compute/unit/hw_unit.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A class for binary vector adder.
 */
class VectorAdderNode : public HardwareUnitNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("isa_list", &isa_list);
  }

  static constexpr const char *_type_key = "ditto.hardware.VectorAdder";
  TVM_DECLARE_FINAL_OBJECT_INFO(VectorAdderNode, HardwareUnitNode);
};

class VectorAdder : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param isa_list The supported isa list
   */
  TVM_DLL VectorAdder(String name, Array<ISA> isa_list);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(VectorAdder, HardwareUnit,
                                        VectorAdderNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VectorAdderNode);
};

} // namespace hardware

} // namespace ditto