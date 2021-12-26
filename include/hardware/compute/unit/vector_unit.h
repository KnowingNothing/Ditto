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
    v->Visit("latency", &latency);
    v->Visit("functionality", &functionality);
  }

  static constexpr const char *_type_key = "ditto.hardware.VectorAdder";
  TVM_DECLARE_FINAL_OBJECT_INFO(VectorAdderNode, HardwareUnitNode);
};

class VectorAdder : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL VectorAdder(String name, double latency,
                      Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(VectorAdder, HardwareUnit,
                                        VectorAdderNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VectorAdderNode);
};

} // namespace hardware

} // namespace ditto