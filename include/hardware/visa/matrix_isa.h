#pragma once

#include <hardware/base/visa_base.h>

using namespace tvm;
namespace ditto {

namespace hardware {

/*!
 * \brief A base class for matrix isa.
 */
class MatrixISANode : public ISANode {
public:
  void VisitAttrs(tvm::AttrVisitor *v) {
    v->Visit("name", &name);
    v->Visit("latency", &latency);
    v->Visit("func", &func);
  }

  static constexpr const char *_type_key = "ditto.hardware.MatrixISA";
  TVM_DECLARE_BASE_OBJECT_INFO(MatrixISANode, ISANode);
};

class MatrixISA : public ISA {
public:
  /*!
   * \brief The constructor.
   * \param name The name of the pattern
   * \param latency The latency of the scalar isa
   * \param func The functionality of the scalar isa
   */
  TVM_DLL MatrixISA(String name, double latency, te::Operation func);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MatrixISA, ISA, MatrixISANode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatrixISANode);
};

MatrixISA
MatrixMultiplyAdd(String name, double latency, runtime::DataType lhs_dtype,
                  runtime::DataType rhs_dtype, runtime::DataType res_dtype,
                  int m, int n, int k, bool left_row_major = true,
                  bool right_row_major = true, bool result_row_major = true);

} // namespace hardware

} // namespace ditto