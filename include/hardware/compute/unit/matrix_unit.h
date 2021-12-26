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
    v->Visit("latency", &latency);
    v->Visit("functionality", &functionality);
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
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL MatrixMultiplyAccumulate(String name, double latency,
                                   Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MatrixMultiplyAccumulate, HardwareUnit,
                                        MatrixMultiplyAccumulateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatrixMultiplyAccumulateNode);
};

} // namespace hardware

} // namespace ditto