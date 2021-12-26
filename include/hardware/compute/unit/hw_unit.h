#include <hardware/compute/hw_compute.h>

namespace ditto {

namespace hardware {

/*!
 * \brief A base class for hardware unit.
 */
class HardwareUnitNode : public HardwareNode {
public:
  double latency;
  Array<te::Operation> functionality;

  static constexpr const char *_type_key = "ditto.hardware.HardwareUnit";
  TVM_DECLARE_BASE_OBJECT_INFO(HardwareUnitNode, HardwareNode);
};

class HardwareUnit : public Hardware {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the hardware
   * \param latency The latency of the hardware
   * \param functionality The functionality of the hardware
   */
  TVM_DLL HardwareUnit(String name, double latency,
                       Array<te::Operation> functionality);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(HardwareUnit, Hardware,
                                        HardwareUnitNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareUnitNode);
};

} // namespace hardware

} // namespace ditto