#pragma once

#include <hardware/base/visa_base.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for scalar isa.
 */
class ScalarISANode : public ISANode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("func", &func);
  }

  static constexpr const char *_type_key = "ditto.hardware.ScalarISA";
  TVM_DECLARE_BASE_OBJECT_INFO(ScalarISANode, ISANode);
};

class ScalarISA : public ISA {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the pattern
   * \param latency The latency of the scalar isa
   * \param func The functionality of the scalar isa
   */
  TVM_DLL ScalarISA(String name, double latency, te::Operation func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScalarISA, ISA, ScalarISANode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScalarISANode);
};

ScalarISA ScalarBinaryAdd(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype);

ScalarISA ScalarBinarySub(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype);

ScalarISA ScalarBinaryMul(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype);

ScalarISA ScalarBinaryDiv(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype);

ScalarISA ScalarBinaryMod(String name, double latency,
                          runtime::DataType lhs_dtype,
                          runtime::DataType rhs_dtype,
                          runtime::DataType res_dtype);

ScalarISA ScalarMultiplyAdd(String name, double latency,
                            runtime::DataType lhs_dtype,
                            runtime::DataType rhs_dtype,
                            runtime::DataType res_dtype);

} // namespace hardware

} // namespace ditto