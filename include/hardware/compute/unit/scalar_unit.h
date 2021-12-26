#include <hardware/compute/unit/hw_unit.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A class for binary adder.
 */
class AdderNode : public HardwareUnitNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("functionality", &functionality);
  }

  static constexpr const char *_type_key = "ditto.hardware.Adder";
  TVM_DECLARE_FINAL_OBJECT_INFO(AdderNode, HardwareUnitNode);
};

class Adder : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL Adder(String name, double latency,
                Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Adder, HardwareUnit, AdderNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AdderNode);
};

/*!
 * \brief A class for binary multiplier.
 */
class MultiplierNode : public HardwareUnitNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("functionality", &functionality);
  }

  static constexpr const char *_type_key = "ditto.hardware.Multiplier";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiplierNode, HardwareUnitNode);
};

class Multiplier : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL Multiplier(String name, double latency,
                     Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Multiplier, HardwareUnit,
                                        MultiplierNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MultiplierNode);
};

/*!
 * \brief A class for binary multiply-adder.
 */
class MultiplyAdderNode : public HardwareUnitNode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("functionality", &functionality);
  }

  static constexpr const char *_type_key = "ditto.hardware.MultiplyAdder";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiplyAdderNode, HardwareUnitNode);
};

class MultiplyAdder : public HardwareUnit {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL MultiplyAdder(String name, double latency,
                        Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MultiplyAdder, HardwareUnit,
                                        MultiplyAdderNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MultiplyAdderNode);
};

} // namespace hardware

} // namespace ditto